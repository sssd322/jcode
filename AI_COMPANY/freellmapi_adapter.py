"""Opt-in, privacy-gated client for the official FreeLLMAPI OpenAI endpoint."""
from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

import free_providers
from jcode_runtime import PermanentRuntimeError, RuntimeUnavailable, TransientRuntimeError

DEFAULT_BASE_URL = "http://127.0.0.1:3001/v1"
DEFAULT_API_KEY_ENV = "VEO_FREELLMAPI_API_KEY"
# Both spellings are the same approved models: the gateway's live /v1/models
# catalog exposes dash-form ids (qwen3-8b) while older configs used Ollama-style
# colon ids.  Only ids listed here may ever be sent to the gateway.
ALLOWED_MODEL_IDS = frozenset(
    {"qwen3:8b", "qwen2.5-coder:3b", "qwen3-8b", "qwen2.5-coder-3b", "llama-3-stheno-8b-v3.2"}
)
DEFAULT_ROLE_MODEL_ALLOWLIST = {
    "planning": ("qwen3:8b",),
    "research": ("qwen3:8b",),
    "qa": ("qwen3:8b",),
    "coding": ("qwen2.5-coder:3b",),
}
_ROLE_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
_ENV_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TRANSIENT_HTTP_CODES = {408, 425, 429}
_KNOWN_CLASSIFICATIONS = {"PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED", "PRIVATE"}
PROBE_MARKER = "VEO_FREELLMAPI_OK"
GATEWAY_SPEC_NAME = "freellmapi"
# The gateway is OpenAI-compatible, so a streaming request would probably work,
# but nothing in this workspace has verified it against the running gateway.
# The adapter therefore sends stream=false and reports the limitation instead of
# pretending streaming is supported.
STREAMING_SUPPORTED = False
STREAMING_REASON = "gateway streaming is unverified; adapter always sends stream=false"
# Tool calling is verified against the live gateway (tools/tool_choice accepted,
# assistant tool_calls round-tripped through the tool role).  The profile is
# deliberately narrow: the adapter never executes anything itself, it can only
# run handlers the caller supplies AND explicitly grants.
TOOL_PROFILE = "lite"
MAX_TOOL_ROUNDS = 3
MAX_TOOL_DEFINITIONS = 16
# Hard caps independent of the model/upstream: one round cannot fan out into
# dozens of handler calls and a run cannot exceed a fixed total budget.
MAX_TOOL_CALLS_PER_ROUND = 4
MAX_TOOL_CALLS_TOTAL = 8
MAX_TOOL_ARGUMENT_CHARS = 20000
MAX_TOOL_RESULT_CHARS = 4000
MAX_TOOL_HANDLER_SECONDS = 30.0
MAX_RESPONSE_BYTES = 4_000_000
_MAX_JSON_DEPTH = 32
_MAX_SCHEMA_DEPTH = 16
# The JSON-schema subset actually enforced.  Validation keywords outside it are
# rejected up front instead of being silently ignored; only pure annotations
# (which carry no constraint) are accepted and ignored.
_SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {"type", "properties", "required", "additionalProperties", "enum", "items"}
)
_ANNOTATION_SCHEMA_KEYWORDS = frozenset({"description", "title", "$comment"})
_SUPPORTED_SCHEMA_TYPES = frozenset(
    {"string", "integer", "number", "boolean", "object", "array", "null"}
)
_TOOL_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_JSON_SCALARS = {"string": str, "boolean": bool, "object": dict, "array": list}


class FreeLLMAPICancelled(TransientRuntimeError):
    """Raised when a caller cancels before or between bounded attempts."""


class FreeLLMAPIToolDenied(PermanentRuntimeError):
    """Raised when a tool call is not allowed to execute (permission/schema)."""


class FreeLLMAPIToolUnsupported(RuntimeUnavailable):
    """Raised when the model does not advertise tool calling in the catalog."""


class FreeLLMAPIToolContractError(PermanentRuntimeError):
    """A tool-call/response contract violation caused by the model or gateway.

    This says nothing about provider health, so it is reported but never counted
    against the shared circuit breaker.
    """


# Quota accounting needs a spec; the gateway is not part of the hosted-free
# allowlist, so it declares its own instead of mutating that registry.
GATEWAY_SPEC = free_providers.FreeProviderSpec(
    name=GATEWAY_SPEC_NAME,
    kind="local_gateway",
    billing_class="LOCAL_GATEWAY",
    base_url=DEFAULT_BASE_URL,
    api_key_env=DEFAULT_API_KEY_ENV,
    models=tuple(sorted(ALLOWED_MODEL_IDS)),
    context_tokens=0,
    requests_per_minute=None,
    requests_per_day=None,
    tokens_per_day=None,
    # The gateway accepts OpenAI `tools`/`tool_choice` and round-trips
    # `tool_calls`; the adapter exposes that only through the gated run_tools()
    # path, never by executing anything itself.
    tools=True,
    trains_on_free_inputs=False,
    recurring_free=True,
    docs_url="https://github.com/Azhan787/FreeLLMAPI",
    notes="Loopback multi-provider gateway; traffic may leave the machine.",
)


class _CountedRuntimeUnavailable(RuntimeUnavailable):
    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


class _CountedPermanentRuntimeError(PermanentRuntimeError):
    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


class _CountedTransientRuntimeError(TransientRuntimeError):
    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


def _is_provider_failure(exc: BaseException) -> bool:
    """Whether an exception means the provider itself is unhealthy.

    Only transport/upstream failures are provider failures.  Tool denials,
    unsupported models, malformed tool calls, invalid arguments and cancellation
    are contract/caller/model failures; counting them on the shared circuit
    breaker would block the healthy text-only path after three bad payloads.
    """
    if isinstance(exc, (FreeLLMAPIToolDenied, FreeLLMAPIToolUnsupported, FreeLLMAPICancelled)):
        return False
    if isinstance(exc, _CountedPermanentRuntimeError):
        return True
    return isinstance(exc, (TransientRuntimeError, RuntimeUnavailable))


@dataclass(frozen=True)
class FreeLLMAPIConfig:
    enabled: bool
    base_url: str
    api_key_env: str
    role_model_allowlist: Mapping[str, tuple[str, ...]]
    allowed_classifications: tuple[str, ...]
    timeout_seconds: float
    max_retries: int
    retry_delay_seconds: float
    max_tokens: int
    requests_per_minute: int = 20
    breaker_failure_threshold: int = 3
    breaker_cooldown_seconds: float = 60.0


def _validate_base_url(value: Any) -> tuple[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base_url must be a non-empty loopback URL")
    parsed = urlsplit(value.strip())
    if parsed.scheme != "http" or parsed.username or parsed.password:
        raise ValueError("base_url must use HTTP on a loopback IP")
    if parsed.path != "/v1" or parsed.query or parsed.fragment:
        raise ValueError("base_url path must be exactly /v1 with no query or fragment")
    host = parsed.hostname or ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("base_url must use a loopback IP literal, not a hostname") from exc
    if not address.is_loopback:
        raise ValueError("base_url must use a loopback IP")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url has an invalid port") from exc
    if port is None or not 1 <= port <= 65535:
        raise ValueError("base_url must include a valid port")
    return value.strip().rstrip("/"), parsed.scheme + "://" + parsed.netloc


def _bounded_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be numeric")
    number = float(value)
    if not minimum <= number <= maximum:
        raise ValueError(name + " must be between " + str(minimum) + " and " + str(maximum))
    return number


def _validated_role_map(role_map: Any) -> dict[str, tuple[str, ...]]:
    # load_config returns a MappingProxyType, so accept any Mapping here or a
    # config that came from disk could never build a client.
    if not isinstance(role_map, Mapping) or not role_map:
        raise ValueError("role_model_allowlist must be a non-empty object")
    normalized: dict[str, tuple[str, ...]] = {}
    for role, models in role_map.items():
        if not isinstance(role, str) or not _ROLE_PATTERN.fullmatch(role):
            raise ValueError("role_model_allowlist contains an invalid role")
        if not isinstance(models, (list, tuple)) or not models:
            raise ValueError("each role must have a non-empty model list")
        if any(not isinstance(model, str) or model not in ALLOWED_MODEL_IDS for model in models):
            raise ValueError("role_model_allowlist may contain only approved model IDs")
        if len(set(models)) != len(models):
            raise ValueError("role model lists must not contain duplicates")
        normalized[role] = tuple(models)
    return normalized


def _validated_tool_definitions(tools: Any, handlers: Mapping[str, Any], granted: Any) -> list[dict]:
    """Normalize OpenAI tool definitions and enforce the permission boundary.

    A definition is acceptable only when the caller registered a handler for it
    AND explicitly granted it.  Nothing is executed here.
    """
    if not isinstance(handlers, Mapping) or not handlers:
        raise FreeLLMAPIToolDenied("tool calling requires a non-empty handler registry")
    if not isinstance(granted, (set, frozenset, list, tuple)):
        raise FreeLLMAPIToolDenied("tool permission must be an explicit allowlist")
    granted_names = {str(name) for name in granted}
    if not granted_names:
        raise FreeLLMAPIToolDenied("no tool permission granted")
    for name, handler in handlers.items():
        if not isinstance(name, str) or not _TOOL_NAME_PATTERN.fullmatch(name):
            raise ValueError("tool handler registry contains an invalid tool name")
        if not callable(handler):
            raise ValueError("tool handler for " + name + " is not callable")
    if not isinstance(tools, (list, tuple)) or not tools:
        raise ValueError("tools must be a non-empty list of tool definitions")
    if len(tools) > MAX_TOOL_DEFINITIONS:
        raise ValueError("at most " + str(MAX_TOOL_DEFINITIONS) + " tool definitions are allowed")
    normalized: list[dict] = []
    seen: set[str] = set()
    for tool in tools:
        if not isinstance(tool, Mapping):
            raise ValueError("each tool definition must be an object")
        function = tool.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("each tool definition needs a function object")
        name = function.get("name")
        if not isinstance(name, str) or not _TOOL_NAME_PATTERN.fullmatch(name):
            raise ValueError("tool definition has an invalid function name")
        if name in seen:
            raise ValueError("tool definitions must not repeat a name")
        seen.add(name)
        if name not in handlers:
            raise FreeLLMAPIToolDenied("tool is not in the handler registry: " + name)
        if name not in granted_names:
            raise FreeLLMAPIToolDenied("tool permission denied: " + name)
        description = function.get("description", "")
        if description is not None and not isinstance(description, str):
            raise ValueError("tool description must be a string")
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, Mapping) or parameters.get("type", "object") != "object":
            raise ValueError("tool parameters must be an object schema")
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description or "",
                    "parameters": dict(parameters),
                },
            }
        )
    return normalized


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    scalar = _JSON_SCALARS.get(expected)
    return scalar is not None and isinstance(value, scalar)


def _json_values_equal(candidate: Any, value: Any) -> bool:
    """JSON-Schema equality that never lets ``True`` match ``1``.

    Numeric int/float equality is kept (JSON has one number type) but booleans
    only match booleans and strings only match strings.
    """
    candidate_is_bool = isinstance(candidate, bool)
    value_is_bool = isinstance(value, bool)
    if candidate_is_bool or value_is_bool:
        return candidate_is_bool and value_is_bool and candidate is value
    if isinstance(candidate, (int, float)) and isinstance(value, (int, float)):
        return candidate == value
    if type(candidate) is not type(value):
        return False
    return candidate == value


def _declared_types(rule: Mapping[str, Any]) -> tuple[str, ...]:
    expected = rule.get("type")
    if expected is None:
        return ()
    if isinstance(expected, str):
        return (expected,)
    if (
        isinstance(expected, (list, tuple))
        and expected
        and all(isinstance(item, str) for item in expected)
        and len(set(expected)) == len(expected)
    ):
        return tuple(str(item) for item in expected)
    raise FreeLLMAPIToolDenied("tool schema has an invalid type declaration")


def _check_schema_node(rule: Any, path: str, depth: int = 0) -> None:
    """Reject anything this adapter cannot enforce instead of ignoring it."""
    if not isinstance(rule, Mapping):
        raise FreeLLMAPIToolDenied("tool schema at " + path + " must be an object")
    if depth > _MAX_SCHEMA_DEPTH:
        raise FreeLLMAPIToolDenied("tool schema at " + path + " is nested too deeply")
    unsupported = sorted(
        str(key)
        for key in set(rule) - _SUPPORTED_SCHEMA_KEYWORDS - _ANNOTATION_SCHEMA_KEYWORDS
    )
    if unsupported:
        raise FreeLLMAPIToolDenied(
            "tool schema at " + path + " uses unsupported keywords: " + ", ".join(unsupported)
        )
    for declared in _declared_types(rule):
        if declared not in _SUPPORTED_SCHEMA_TYPES:
            raise FreeLLMAPIToolDenied(
                "tool schema at " + path + " has unsupported type: " + declared
            )
    properties = rule.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise FreeLLMAPIToolDenied("tool schema at " + path + ".properties must be an object")
        for child_name, child in properties.items():
            if not isinstance(child_name, str):
                raise FreeLLMAPIToolDenied(
                    "tool schema at " + path + ".properties has a non-string key"
                )
            _check_schema_node(child, path + ".properties." + child_name, depth + 1)
    required = rule.get("required")
    if required is not None and (
        not isinstance(required, (list, tuple))
        or any(not isinstance(item, str) for item in required)
    ):
        raise FreeLLMAPIToolDenied(
            "tool schema at " + path + ".required must be a list of strings"
        )
    additional = rule.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool):
        raise FreeLLMAPIToolDenied(
            "tool schema at " + path + " must use a boolean additionalProperties"
        )
    enum = rule.get("enum")
    if enum is not None and (not isinstance(enum, (list, tuple)) or not enum):
        raise FreeLLMAPIToolDenied("tool schema at " + path + ".enum must be a non-empty list")
    items = rule.get("items")
    if items is not None:
        _check_schema_node(items, path + ".items", depth + 1)


def _json_depth(value: Any) -> int:
    """Nesting depth of a parsed JSON value, computed iteratively."""
    deepest = 1
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > deepest:
            deepest = depth
        if isinstance(node, dict):
            for child in node.values():
                stack.append((child, depth + 1))
        elif isinstance(node, list):
            for child in node:
                stack.append((child, depth + 1))
    return deepest


def _validate_tool_arguments(schema: Mapping[str, Any], arguments: Any) -> dict:
    """Validate arguments against the declared JSON-schema subset, strictly.

    Supported keywords: type, properties, required, additionalProperties, enum,
    items.  Unknown keywords are rejected; mismatches are refused, never coerced.
    """
    _check_schema_node(schema, chr(36))
    if not isinstance(arguments, dict):
        raise FreeLLMAPIToolDenied("tool arguments must be a JSON object")
    properties = schema.get("properties") or {}
    if not isinstance(properties, Mapping):
        raise FreeLLMAPIToolDenied("tool schema properties must be an object")
    required = schema.get("required") or []
    if not isinstance(required, (list, tuple)):
        raise FreeLLMAPIToolDenied("tool schema required must be a list")
    for key in required:
        if key not in arguments:
            raise FreeLLMAPIToolDenied("tool arguments are missing required field: " + str(key))
    if schema.get("additionalProperties", True) is False:
        extra = set(arguments) - set(properties)
        if extra:
            raise FreeLLMAPIToolDenied(
                "tool arguments contain undeclared fields: " + ", ".join(sorted(extra))
            )
    for key, value in arguments.items():
        rule = properties.get(key)
        if rule is None:
            continue
        if not isinstance(rule, Mapping):
            raise FreeLLMAPIToolDenied("tool schema for " + str(key) + " is invalid")
        expected = rule.get("type")
        if isinstance(expected, str) and not _json_type_matches(value, expected):
            raise FreeLLMAPIToolDenied(
                "tool argument " + str(key) + " is not of type " + expected
            )
        allowed = rule.get("enum")
        if isinstance(allowed, (list, tuple)) and value not in allowed:
            raise FreeLLMAPIToolDenied("tool argument " + str(key) + " is not an allowed value")
        if expected == "array" and isinstance(value, list):
            items = rule.get("items")
            if isinstance(items, Mapping) and isinstance(items.get("type"), str):
                for item in value:
                    if not _json_type_matches(item, str(items["type"])):
                        raise FreeLLMAPIToolDenied(
                            "tool argument " + str(key) + " contains a wrongly typed item"
                        )
    return dict(arguments)


def load_config(path: str | Path) -> FreeLLMAPIConfig:
    """Load a strict config; secrets are never accepted from the JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("FreeLLMAPI configuration must be a JSON object")
    allowed_fields = {
        "enabled",
        "base_url",
        "api_key_env",
        "role_model_allowlist",
        "allowed_classifications",
        "timeout_seconds",
        "max_retries",
        "retry_delay_seconds",
        "max_tokens",
        "requests_per_minute",
        "breaker_failure_threshold",
        "breaker_cooldown_seconds",
    }
    unknown_fields = set(payload) - allowed_fields
    if unknown_fields:
        raise ValueError("unsupported FreeLLMAPI configuration fields: " + ", ".join(sorted(unknown_fields)))

    enabled = payload.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("enabled must be a boolean")
    base_url, _ = _validate_base_url(payload.get("base_url", DEFAULT_BASE_URL))
    api_key_env = payload.get("api_key_env", DEFAULT_API_KEY_ENV)
    if not isinstance(api_key_env, str) or not _ENV_PATTERN.fullmatch(api_key_env):
        raise ValueError("api_key_env must be an environment variable name")

    default_roles = {role: list(models) for role, models in DEFAULT_ROLE_MODEL_ALLOWLIST.items()}
    roles = _validated_role_map(payload.get("role_model_allowlist", default_roles))
    classifications = payload.get("allowed_classifications", ["PUBLIC"])
    if classifications != ["PUBLIC"]:
        raise ValueError("allowed_classifications must be exactly [\"PUBLIC\"]")

    timeout_seconds = _bounded_number(payload.get("timeout_seconds", 45), "timeout_seconds", 1, 120)
    retry_delay_seconds = _bounded_number(
        payload.get("retry_delay_seconds", 0.25), "retry_delay_seconds", 0, 2
    )
    max_retries = payload.get("max_retries", 1)
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 2:
        raise ValueError("max_retries must be an integer between 0 and 2")
    max_tokens = payload.get("max_tokens", 512)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 2048:
        raise ValueError("max_tokens must be an integer between 1 and 2048")
    requests_per_minute = payload.get("requests_per_minute", 20)
    if (
        isinstance(requests_per_minute, bool)
        or not isinstance(requests_per_minute, int)
        or not 1 <= requests_per_minute <= 600
    ):
        raise ValueError("requests_per_minute must be an integer between 1 and 600")
    breaker_failure_threshold = payload.get("breaker_failure_threshold", 3)
    if (
        isinstance(breaker_failure_threshold, bool)
        or not isinstance(breaker_failure_threshold, int)
        or not 1 <= breaker_failure_threshold <= 10
    ):
        raise ValueError("breaker_failure_threshold must be an integer between 1 and 10")
    breaker_cooldown_seconds = _bounded_number(
        payload.get("breaker_cooldown_seconds", 60), "breaker_cooldown_seconds", 1, 3600
    )

    return FreeLLMAPIConfig(
        enabled=enabled,
        base_url=base_url,
        api_key_env=api_key_env,
        role_model_allowlist=MappingProxyType(roles),
        allowed_classifications=("PUBLIC",),
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        max_tokens=max_tokens,
        requests_per_minute=requests_per_minute,
        breaker_failure_threshold=breaker_failure_threshold,
        breaker_cooldown_seconds=breaker_cooldown_seconds,
    )


def _open_direct(request: urllib.request.Request, timeout: float):
    """Bypass environment proxies; requests are deliberately loopback-only."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


class FreeLLMAPIClient:
    """Stateless text adapter; it is not wired into JCode's worker runtime."""

    provider = "freellmapi"
    class_ = "OPTIONAL_GATEWAY"

    def __init__(
        self,
        config: FreeLLMAPIConfig,
        telemetry: Callable[[dict[str, Any]], None] | None = None,
        quota: "free_providers.QuotaLedger | None" = None,
        breaker: "free_providers.CircuitBreaker | None" = None,
    ):
        if not config.enabled:
            raise RuntimeUnavailable("FreeLLMAPI is disabled")
        self.base_url, self.origin = _validate_base_url(config.base_url)
        if config.allowed_classifications != ("PUBLIC",):
            raise ValueError("FreeLLMAPI may accept PUBLIC data only")
        self.role_model_allowlist = MappingProxyType(_validated_role_map(config.role_model_allowlist))
        if not isinstance(config.api_key_env, str) or not _ENV_PATTERN.fullmatch(config.api_key_env):
            raise ValueError("api_key_env must be an environment variable name")
        self.api_key_env = config.api_key_env
        self.timeout_seconds = _bounded_number(config.timeout_seconds, "timeout_seconds", 1, 120)
        if isinstance(config.max_retries, bool) or not isinstance(config.max_retries, int):
            raise ValueError("max_retries must be an integer")
        if not 0 <= config.max_retries <= 2:
            raise ValueError("max_retries must be between 0 and 2")
        self.max_retries = config.max_retries
        self.retry_delay_seconds = _bounded_number(
            config.retry_delay_seconds, "retry_delay_seconds", 0, 2
        )
        if isinstance(config.max_tokens, bool) or not isinstance(config.max_tokens, int):
            raise ValueError("max_tokens must be an integer")
        if not 1 <= config.max_tokens <= 2048:
            raise ValueError("max_tokens must be between 1 and 2048")
        self.max_tokens = config.max_tokens
        self.telemetry = telemetry
        # Quota and failure isolation reuse the registry's own accounting
        # objects.  The gateway keeps its spec here because PROVIDER_SPECS is
        # the hosted-free allowlist and the gateway is a loopback gateway.
        self.spec = replace(
            GATEWAY_SPEC, requests_per_minute=int(config.requests_per_minute)
        )
        self.quota = quota if quota is not None else free_providers.QuotaLedger()
        self.breaker = breaker if breaker is not None else free_providers.CircuitBreaker(
            failure_threshold=int(config.breaker_failure_threshold),
            cooldown_seconds=float(config.breaker_cooldown_seconds),
        )
        self.supports_streaming = STREAMING_SUPPORTED
        # Measured latency only; None means "never observed", never a guess.
        self._latency_ema: float | None = None
        self.last_probe: dict | None = None

    def descriptor(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "class": self.class_,
            "base_url": self.base_url,
            "api_key_source": "environment",
            "api_key_env": self.api_key_env,
            "role_model_allowlist": {
                role: list(models) for role, models in sorted(self.role_model_allowlist.items())
            },
            "allowed_classifications": ["PUBLIC"],
            "tool_profile": TOOL_PROFILE,
            "tool_calling": {
                "supported": True,
                "max_rounds": MAX_TOOL_ROUNDS,
                "max_tools": MAX_TOOL_DEFINITIONS,
                "permission_required": True,
                "schema_validated": True,
                "capability_source": "/v1/models supported_parameters",
            },
            "paid": False,
            "billing_class": self.spec.billing_class,
            "requests_per_minute": self.spec.requests_per_minute,
            "streaming": STREAMING_SUPPORTED,
            "streaming_reason": STREAMING_REASON,
            "circuit": self.breaker.state_of(self.provider),
            "quota": self.quota.snapshot(self.provider),
        }

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeUnavailable(
                "FreeLLMAPI requires the unified API key in environment variable "
                + self.api_key_env
            )
        if "\r" in key or "\n" in key:
            raise PermanentRuntimeError("FreeLLMAPI API key environment value is invalid")
        return key

    def _request(
        self,
        path: str,
        payload: dict[str, Any] | None,
        timeout: float,
        authenticated: bool,
        cancel: "threading.Event | None" = None,
    ) -> tuple[dict[str, Any], Any, int, int]:
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ValueError("FreeLLMAPI request path must be an absolute fixed path")
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if authenticated:
            headers["Authorization"] = "Bearer " + self._api_key()
        request = urllib.request.Request(
            self.origin + path,
            data=data,
            headers=headers,
            method="GET" if data is None else "POST",
        )

        for attempt in range(1, self.max_retries + 2):
            if cancel is not None and cancel.is_set():
                raise FreeLLMAPICancelled(
                    "FreeLLMAPI request cancelled before attempt " + str(attempt)
                )
            try:
                with _open_direct(request, timeout) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    response_headers = getattr(response, "headers", {})
                    status = int(getattr(response, "status", 200))
            except urllib.error.HTTPError as exc:
                message = "FreeLLMAPI HTTP " + str(exc.code)
                if int(exc.code) == 429:
                    self.quota.note_rate_limited(self.provider, "HTTP 429")
                retryable = exc.code in _TRANSIENT_HTTP_CODES or 500 <= int(exc.code) < 600
                if retryable and attempt <= self.max_retries:
                    time.sleep(self.retry_delay_seconds * attempt)
                    continue
                if retryable:
                    raise _CountedTransientRuntimeError(message, attempt) from None
                raise _CountedPermanentRuntimeError(message, attempt) from None
            except (socket.timeout, TimeoutError, urllib.error.URLError, OSError) as exc:
                if attempt <= self.max_retries:
                    time.sleep(self.retry_delay_seconds * attempt)
                    continue
                if isinstance(exc, (socket.timeout, TimeoutError)):
                    message = "FreeLLMAPI timed out after " + str(timeout) + "s"
                else:
                    message = "FreeLLMAPI loopback transport failed: " + type(exc).__name__
                raise _CountedTransientRuntimeError(message, attempt) from None

            try:
                result = json.loads(body)
            except json.JSONDecodeError:
                raise _CountedPermanentRuntimeError(
                    "FreeLLMAPI returned invalid JSON", attempt
                ) from None
            if not isinstance(result, dict):
                raise _CountedPermanentRuntimeError(
                    "FreeLLMAPI returned a non-object response", attempt
                )
            return result, response_headers, attempt, status

        raise AssertionError("bounded FreeLLMAPI retry loop exhausted unexpectedly")

    def _guard_dispatch(
        self, request_id: str, role: str | None, model: str | None, started: float
    ) -> None:
        """Quota and breaker gate, checked before any network access."""
        can_call, quota_reason = self.quota.can_call(self.spec)
        if not can_call:
            self._emit(
                {
                    "provider": self.provider,
                    "operation": "chat.completions",
                    "role": role,
                    "model": model,
                    "classification": "PUBLIC",
                    "outcome": "quota_exhausted",
                    "reason": quota_reason,
                    "request_id": request_id,
                    "attempts": 0,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                }
            )
            raise RuntimeUnavailable("FreeLLMAPI free quota exhausted: " + quota_reason)
        if not self.breaker.allow(self.provider):
            self._emit(
                {
                    "provider": self.provider,
                    "operation": "chat.completions",
                    "role": role,
                    "model": model,
                    "classification": "PUBLIC",
                    "outcome": "circuit_open",
                    "request_id": request_id,
                    "attempts": 0,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                }
            )
            raise TransientRuntimeError("FreeLLMAPI circuit is open after repeated failures")

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        timeout: float,
        token_limit: int,
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        request_id: str | None = None,
        cancel: "threading.Event | None" = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int, int]:
        """One /v1/chat/completions round trip; returns the assistant message."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": float(temperature),
            "max_tokens": token_limit,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        response, _, attempts, status = self._request(
            "/v1/chat/completions",
            payload,
            timeout,
            authenticated=True,
            cancel=cancel,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise PermanentRuntimeError("FreeLLMAPI response has no completion choice")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise PermanentRuntimeError("FreeLLMAPI response has no assistant message")
        return message, response, choices[0], attempts, status

    def _catalog_rows(
        self, timeout: float, cancel: "threading.Event | None" = None
    ) -> tuple[list[dict[str, Any]], int]:
        payload, _, attempts, _ = self._request(
            "/v1/models",
            None,
            min(timeout, self.timeout_seconds),
            authenticated=True,
            cancel=cancel,
        )
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise RuntimeUnavailable("FreeLLMAPI /v1/models response has no data list")
        return [row for row in rows if isinstance(row, dict)], attempts

    @staticmethod
    def _model_tool_disclosure(rows: list[dict[str, Any]], model: str) -> bool | None:
        """Whether the catalog advertises `tools` for this exact model id.

        Returns None when the model is absent, so "unknown" is never reported as
        "supported".
        """
        for row in rows:
            if row.get("id") != model:
                continue
            advertised = row.get("supported_parameters")
            if not isinstance(advertised, list):
                return None
            return "tools" in advertised
        return None

    @staticmethod
    def _parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
        """Strictly parse assistant tool_calls; never guess or repair them."""
        raw = message.get("tool_calls")
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise PermanentRuntimeError("FreeLLMAPI returned malformed tool_calls")
        if not raw:
            # Some gateways serialize an empty list instead of omitting it.
            return []
        calls: list[dict[str, Any]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise PermanentRuntimeError("FreeLLMAPI returned a malformed tool call")
            call_id = entry.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise PermanentRuntimeError("FreeLLMAPI tool call has no id")
            kind = entry.get("type", "function")
            if kind != "function":
                raise PermanentRuntimeError("FreeLLMAPI tool call has an unsupported type")
            function = entry.get("function")
            if not isinstance(function, dict):
                raise PermanentRuntimeError("FreeLLMAPI tool call has no function object")
            name = function.get("name")
            if not isinstance(name, str) or not _TOOL_NAME_PATTERN.fullmatch(name):
                raise PermanentRuntimeError("FreeLLMAPI tool call has an invalid name")
            arguments = function.get("arguments")
            if isinstance(arguments, dict):
                # Some gateways pre-parse; keep the original id but re-serialize.
                arguments = json.dumps(arguments, ensure_ascii=False)
            if not isinstance(arguments, str):
                raise PermanentRuntimeError("FreeLLMAPI tool call arguments are not a string")
            try:
                parsed = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                raise PermanentRuntimeError(
                    "FreeLLMAPI tool call arguments are not valid JSON"
                ) from None
            if not isinstance(parsed, dict):
                raise PermanentRuntimeError("FreeLLMAPI tool call arguments are not an object")
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        return calls

    def run_tools(
        self,
        prompt: str,
        role: str,
        model: str,
        *,
        tools: Any,
        handlers: Mapping[str, Callable[[dict[str, Any]], Any]],
        granted: Any,
        data_classification: str = "PRIVATE",
        source: str = "company",
        timeout: float | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        max_rounds: int = MAX_TOOL_ROUNDS,
        tool_choice: str = "auto",
        request_id: str | None = None,
        cancel: "threading.Event | None" = None,
        fallback_to_text: bool = False,
    ) -> dict[str, Any]:
        """Bounded, permission-gated tool calling over the gateway.

        The adapter never executes a tool by itself: every call must name a
        caller-supplied handler that the caller also explicitly granted.  Tool
        arguments are validated against the declared schema before execution.
        """
        started = time.monotonic()
        role = str(role or "").strip()
        model = str(model or "").strip()
        classification = str(data_classification or "").strip().upper()
        source = str(source or "").strip().lower()
        if classification != "PUBLIC" or source != "public_reference":
            raise PermanentRuntimeError("FreeLLMAPI blocks non-public or non-public-reference data")
        allowed_models = self.role_model_allowlist.get(role)
        if allowed_models is None:
            raise PermanentRuntimeError("role is not in the FreeLLMAPI allowlist")
        if model not in allowed_models:
            raise PermanentRuntimeError("model is not allowlisted for this FreeLLMAPI role")
        if not isinstance(prompt, str) or not prompt.strip():
            raise PermanentRuntimeError("prompt is required")
        if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or not 1 <= max_rounds <= 5:
            raise ValueError("max_rounds must be an integer between 1 and 5")
        if tool_choice not in {"auto", "required", "none"}:
            raise ValueError("tool_choice must be auto, required or none")
        definitions = _validated_tool_definitions(tools, handlers, granted)
        schemas = {
            definition["function"]["name"]: definition["function"].get("parameters") or {}
            for definition in definitions
        }

        request_timeout = self.timeout_seconds
        if timeout is not None:
            request_timeout = min(self.timeout_seconds, _bounded_number(timeout, "timeout", 1, 3600))
        token_limit = self.max_tokens if max_tokens is None else max_tokens
        if isinstance(token_limit, bool) or not isinstance(token_limit, int):
            raise PermanentRuntimeError("max_tokens must be an integer")
        if not 1 <= token_limit <= self.max_tokens:
            raise PermanentRuntimeError("max_tokens exceeds the FreeLLMAPI limit")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise PermanentRuntimeError("temperature must be numeric")
        if not 0 <= float(temperature) <= 2:
            raise PermanentRuntimeError("temperature must be between 0 and 2")
        if cancel is not None and cancel.is_set():
            raise FreeLLMAPICancelled("FreeLLMAPI request cancelled before dispatch")
        request_id = str(request_id or uuid.uuid4().hex)
        self._guard_dispatch(request_id, role, model, started)

        deadline = started + request_timeout
        attempts = 0
        try:
            rows, discovery_attempts = self._catalog_rows(request_timeout, cancel=cancel)
            attempts += discovery_attempts
            disclosed = self._model_tool_disclosure(rows, model)
            if disclosed is not True:
                if not fallback_to_text:
                    raise FreeLLMAPIToolUnsupported(
                        "model does not advertise tool calling in the FreeLLMAPI catalog"
                    )
                text_result = self.chat(
                    prompt,
                    role,
                    model,
                    data_classification="PUBLIC",
                    source="public_reference",
                    timeout=timeout,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    request_id=request_id,
                    cancel=cancel,
                )
                text_result.update(
                    {
                        "tool_profile": "none",
                        "tool_calling": "unsupported_by_model",
                        "tool_rounds": 0,
                        "tool_calls": [],
                        "tools_executed": 0,
                    }
                )
                return text_result

            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
            executed: list[dict[str, Any]] = []
            rounds = 0
            text = ""
            exhausted = False
            status = 0
            while True:
                if cancel is not None and cancel.is_set():
                    raise FreeLLMAPICancelled("FreeLLMAPI tool loop cancelled")
                remaining = deadline - time.monotonic()
                if remaining < 1:
                    raise _CountedTransientRuntimeError(
                        "FreeLLMAPI timed out after " + str(request_timeout) + "s", attempts
                    )
                message, response, choice, request_attempts, status = self._chat_completion(
                    messages,
                    model,
                    remaining,
                    token_limit,
                    temperature,
                    tools=definitions,
                    # The choice applies to the first round only: after a tool
                    # result the model must be free to answer.
                    tool_choice=(tool_choice if rounds == 0 and tool_choice != "auto" else None),
                    request_id=request_id,
                    cancel=cancel,
                )
                attempts += request_attempts
                calls = self._parse_tool_calls(message)
                content = message.get("content")
                if not calls:
                    text = content if isinstance(content, str) else ""
                    break
                if rounds >= max_rounds:
                    # Budget spent: show the model its own pending request once
                    # more with tools disabled, and discard the calls unexecuted.
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content if isinstance(content, str) else None,
                            "tool_calls": calls,
                        }
                    )
                    final_message, response, choice, request_attempts, status = self._chat_completion(
                        messages,
                        model,
                        max(1.0, deadline - time.monotonic()),
                        token_limit,
                        temperature,
                        tools=definitions,
                        tool_choice="none",
                        request_id=request_id,
                        cancel=cancel,
                    )
                    attempts += request_attempts
                    pending = self._parse_tool_calls(final_message)
                    final_content = final_message.get("content")
                    text = final_content if isinstance(final_content, str) else ""
                    exhausted = bool(pending) or not text.strip()
                    break
                messages.append(
                    {
                        "role": "assistant",
                        "content": content if isinstance(content, str) else None,
                        "tool_calls": calls,
                    }
                )
                for call in calls:
                    name = call["function"]["name"]
                    if name not in schemas or name not in handlers:
                        raise FreeLLMAPIToolDenied("tool permission denied: " + name)
                    arguments = _validate_tool_arguments(
                        schemas[name], json.loads(call["function"]["arguments"] or "{}")
                    )
                    try:
                        outcome = handlers[name](arguments)
                        error = None
                    except Exception as exc:  # a failing tool must not kill the run
                        outcome = None
                        error = type(exc).__name__ + ": " + str(exc)[:200]
                    rendered = json.dumps(
                        {"ok": error is None, "result": outcome, "error": error},
                        ensure_ascii=False,
                        default=str,
                    )
                    if len(rendered) > MAX_TOOL_RESULT_CHARS:
                        rendered = rendered[:MAX_TOOL_RESULT_CHARS]
                    executed.append(
                        {
                            "id": call["id"],
                            "name": name,
                            "arguments": json.loads(call["function"]["arguments"] or "{}"),
                            "ok": error is None,
                            "error": error,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": name,
                            "content": rendered,
                        }
                    )
                rounds += 1
    
            if not text.strip():
                raise PermanentRuntimeError(
                    "FreeLLMAPI returned an empty assistant answer"
                    + (" after the tool budget" if exhausted else "")
                )
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            total_tokens = usage.get("total_tokens")
            self.quota.note_call(
                self.provider,
                int(total_tokens) if isinstance(total_tokens, (int, float)) else 0,
            )
            self.breaker.record_success(self.provider)
            self._note_latency(time.monotonic() - started)
            self._emit(
                {
                    "provider": self.provider,
                    "operation": "chat.completions.tools",
                    "role": role,
                    "model": model,
                    "classification": "PUBLIC",
                    "outcome": "success",
                    "tool_rounds": rounds,
                    "tool_calls": len(executed),
                    "tools": sorted({entry["name"] for entry in executed}),
                    "request_id": request_id,
                    "attempts": attempts,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                }
            )
            return {
                "text": text,
                "provider": self.provider,
                "model": model,
                "response_model": (
                    response.get("model") if isinstance(response.get("model"), str) else None
                ),
                "session_id": (
                    response.get("id") if isinstance(response.get("id"), str) else None
                ),
                "usage": usage,
                "latency_seconds": round(time.monotonic() - started, 3),
                "http_status": status,
                "attempts": attempts,
                "request_id": request_id,
                "data_classification": "PUBLIC",
                "transport": "freellmapi-openai-compatible",
                "quota": self.quota.snapshot(self.provider),
                "tool_profile": TOOL_PROFILE,
                "tool_calling": "verified",
                "tool_rounds": rounds,
                "tool_calls": executed,
                "tools_executed": len(executed),
                "tool_budget_exhausted": exhausted,
            }
        except (PermanentRuntimeError, RuntimeUnavailable, TransientRuntimeError) as exc:
            attempts += getattr(exc, "attempts", 0)
            if not isinstance(exc, FreeLLMAPICancelled):
                self.breaker.record_failure(
                    self.provider, type(exc).__name__ + ": " + str(exc)[:150]
                )
            self._emit(
                {
                    "provider": self.provider,
                    "operation": "chat.completions.tools",
                    "role": role,
                    "model": model,
                    "classification": "PUBLIC",
                    "outcome": "error",
                    "error_type": type(exc).__name__,
                    "request_id": request_id,
                    "attempts": attempts,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                }
            )
            raise

    def health(self, timeout: float = 5) -> dict[str, Any]:
        payload, _, _, _ = self._request(
            # FreeLLMAPI exposes its unauthenticated container liveness probe
            # here; `/health` is not part of its deployed HTTP surface.
            "/api/ping",
            None,
            min(_bounded_number(timeout, "timeout", 1, 120), self.timeout_seconds),
            authenticated=False,
        )
        if payload.get("status") != "ok":
            raise RuntimeUnavailable("FreeLLMAPI /api/ping response is not ok")
        return payload

    @staticmethod
    def _model_is_ready(row: dict[str, Any]) -> bool:
        status = row.get("execution_status", row.get("executionStatus"))
        if status is not None:
            return status == "ready"
        available = row.get("available")
        return available is True or available == 1

    def _list_models(
        self, timeout: float, cancel: "threading.Event | None" = None
    ) -> tuple[list[str], int]:
        payload, _, attempts, _ = self._request(
            "/v1/models",
            None,
            min(timeout, self.timeout_seconds),
            authenticated=True,
            cancel=cancel,
        )
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise _CountedRuntimeUnavailable(
                "FreeLLMAPI /v1/models response has no data list", attempts
            )
        models = {
            row["id"]
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("id"), str)
            and self._model_is_ready(row)
            and row["id"] in ALLOWED_MODEL_IDS
        }
        return sorted(models), attempts

    def list_models(self, timeout: float = 8) -> list[str]:
        timeout = _bounded_number(timeout, "timeout", 1, 120)
        models, _ = self._list_models(timeout)
        return models

    @staticmethod
    def _upstream_of(row: dict[str, Any]) -> str:
        """Which backend would serve this model row.

        The gateway fans a single upstream (for example AI Horde) out into many
        model ids.  Grouping by upstream keeps the fleet from believing it has
        dozens of independent providers when it has one.
        """
        platforms = row.get("platforms")
        if isinstance(platforms, list):
            names = sorted({str(item).strip() for item in platforms if str(item).strip()})
            if names:
                return ",".join(names)
        owned = row.get("owned_by")
        if isinstance(owned, str) and owned.strip():
            return owned.strip()
        return "unknown"

    def catalog(self, timeout: float = 8) -> dict[str, Any]:
        """Read the gateway's real `/v1/models` and group rows by upstream."""
        timeout = _bounded_number(timeout, "timeout", 1, 120)
        payload, _, attempts, status = self._request(
            "/v1/models",
            None,
            min(timeout, self.timeout_seconds),
            authenticated=True,
        )
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise RuntimeUnavailable("FreeLLMAPI /v1/models response has no data list")
        models: list[dict[str, Any]] = []
        upstreams: dict[str, list[str]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                continue
            model_id = row["id"]
            upstream = self._upstream_of(row)
            ready = self._model_is_ready(row)
            models.append(
                {
                    "id": model_id,
                    "upstream": upstream,
                    "ready": ready,
                    "execution_status": row.get("execution_status", row.get("executionStatus")),
                    "allowed": model_id in ALLOWED_MODEL_IDS,
                }
            )
            upstreams.setdefault(upstream, []).append(model_id)
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "http_status": status,
            "attempts": attempts,
            "model_count": len(models),
            "ready_count": sum(1 for entry in models if entry["ready"]),
            "backend_count": len(upstreams),
            "upstreams": {name: sorted(ids) for name, ids in sorted(upstreams.items())},
            "models": models,
            "allowed_ready_models": sorted(
                entry["id"] for entry in models if entry["ready"] and entry["allowed"]
            ),
        }

    def _role_for_model(self, model: str) -> str | None:
        """Least-privilege role whose allowlist contains ``model``."""
        matches = [
            role for role, models in self.role_model_allowlist.items() if model in models
        ]
        if not matches:
            return None
        for preferred in ("coding", "qa", "research", "planning"):
            if preferred in matches:
                return preferred
        return sorted(matches)[0]

    # ---------------------------------------------------------- worker contract
    def execute(
        self,
        prompt: str,
        workspace: Path | str,
        model: str | None = None,
        timeout: float | None = None,
        tool_profile: str | None = None,
        max_tokens: int | None = None,
        data_classification: str = "PRIVATE",
        request_id: str | None = None,
        cancel: "threading.Event | None" = None,
        tools: Any = None,
        handlers: Mapping[str, Callable[[dict[str, Any]], Any]] | None = None,
        granted: Any = None,
    ) -> dict[str, Any]:
        """Worker-facing entry point; fails closed for anything not PUBLIC.

        ``data_classification`` defaults to PRIVATE so a caller that forgets to
        classify can never leak company text to the gateway.
        """
        classification = str(data_classification or "").strip().upper()
        if classification != "PUBLIC":
            raise PermanentRuntimeError(
                "FreeLLMAPI may receive PUBLIC data only; task is "
                + (classification or "UNSPECIFIED")
            )
        Path(workspace).mkdir(parents=True, exist_ok=True)
        prompt = str(prompt or "").strip()
        if not prompt:
            raise PermanentRuntimeError("prompt is required")
        selected = str(model or "").strip()
        role = self._role_for_model(selected) if selected else None
        if role is None and not selected:
            role = (
                "research"
                if "research" in self.role_model_allowlist
                else sorted(self.role_model_allowlist)[0]
            )
        if role is None:
            raise PermanentRuntimeError("model is not allowlisted for any FreeLLMAPI role")
        if tools is not None:
            tool_result = self.run_tools(
                prompt,
                role,
                selected or self.role_model_allowlist[role][0],
                tools=tools,
                handlers=handlers or {},
                granted=granted,
                data_classification="PUBLIC",
                source="public_reference",
                timeout=timeout,
                max_tokens=max_tokens,
                request_id=request_id,
                cancel=cancel,
            )
            tool_result.update(
                {
                    "class": self.class_,
                    "paid": False,
                    "exit_code": 0,
                    "data_classification": "PUBLIC",
                    "streaming": STREAMING_SUPPORTED,
                }
            )
            return tool_result
        result = self.chat(
            prompt,
            role,
            selected or self.role_model_allowlist[role][0],
            data_classification="PUBLIC",
            source="public_reference",
            timeout=timeout,
            max_tokens=max_tokens,
            request_id=request_id,
            cancel=cancel,
        )
        result.update(
            {
                "class": self.class_,
                "paid": False,
                "tool_profile": "none",
                "exit_code": 0,
                "data_classification": "PUBLIC",
                "streaming": STREAMING_SUPPORTED,
            }
        )
        return result

    def probe(
        self,
        model: str | None = None,
        workspace: Path | str = ".",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Real PUBLIC smoke prompt; a missing key is BLOCKED, not healthy."""
        started = time.monotonic()
        result: dict[str, Any] = {
            "provider": self.provider,
            "class": self.class_,
            "paid": False,
            "model": model,
            "models": [],
            "available": False,
            "ok": False,
            "marker_ok": False,
            "reason": "",
            "error": None,
            "base_url": self.base_url,
            "circuit": self.breaker.state_of(self.provider),
            "catalog": None,
            "checked_at": free_providers.now_iso(),
            "latency_seconds": None,
            "smoke_text": "",
            "streaming": STREAMING_SUPPORTED,
        }
        budget = min(_bounded_number(timeout or self.timeout_seconds, "timeout", 1, 3600), self.timeout_seconds)
        try:
            catalog = self.catalog(timeout=min(8.0, budget))
            result["catalog"] = {
                key: catalog[key]
                for key in ("model_count", "ready_count", "backend_count", "upstreams")
            }
            models = list(catalog["allowed_ready_models"])
            result["models"] = models
            selected = str(model or "").strip() or (models[0] if models else "")
            result["model"] = selected or None
            if not selected:
                raise RuntimeUnavailable("no allowlisted model is ready in FreeLLMAPI")
            # The smoke prompt is a fixed literal with no company data, so it is
            # PUBLIC by construction; execute() still fails closed by default.
            smoke = self.execute(
                "Reply with exactly " + PROBE_MARKER + " and nothing else.",
                workspace,
                selected,
                timeout=budget,
                max_tokens=min(32, self.max_tokens),
                data_classification="PUBLIC",
            )
            text = str(smoke.get("text") or "")
            result["smoke_text"] = text[:300]
            result["marker_ok"] = PROBE_MARKER in text
            result["available"] = True
            result["ok"] = True
            result["reason"] = (
                "FreeLLMAPI answered with " + PROBE_MARKER
                if result["marker_ok"]
                else "FreeLLMAPI answered without the readiness marker"
            )
        except Exception as exc:
            result["error"] = type(exc).__name__ + ": " + str(exc)[:300]
            result["reason"] = result["error"]
        result["latency_seconds"] = round(time.monotonic() - started, 2)
        result["circuit"] = self.breaker.state_of(self.provider)
        self.last_probe = dict(result)
        return result

    @property
    def credential_present(self) -> bool:
        """Whether a key exists, without ever reading or returning its value."""
        return bool(os.environ.get(self.api_key_env, "").strip())

    @property
    def latency_seconds(self) -> float | None:
        """Exponentially weighted latency of real calls, or None if unmeasured."""
        return self._latency_ema

    def _note_latency(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self._latency_ema is None:
            self._latency_ema = round(float(seconds), 3)
        else:
            self._latency_ema = round(0.7 * self._latency_ema + 0.3 * float(seconds), 3)

    def route_status(self) -> dict[str, Any]:
        """Cheap, network-free routing view used by ``ProviderFleet.routes``.

        A gateway whose free quota is exhausted or whose breaker is OPEN must not
        appear as a route: routing reflects health, not just the catalog.
        """
        can_call, quota_reason = self.quota.can_call(self.spec)
        circuit = self.breaker.state_of(self.provider)
        routable = bool(can_call) and circuit != free_providers.BREAKER_OPEN
        return {
            "provider": self.provider,
            "routable": routable,
            "reason": None if routable else (quota_reason if not can_call else "CIRCUIT_OPEN"),
            "circuit": circuit,
            "quota": self.quota.snapshot(self.provider),
            "latency_seconds": self.latency_seconds,
            "streaming": STREAMING_SUPPORTED,
        }

    def _emit(self, event: dict[str, Any]) -> None:
        if self.telemetry is not None:
            self.telemetry(dict(event))

    def chat(
        self,
        prompt: str,
        role: str,
        model: str,
        *,
        data_classification: str = "PRIVATE",
        source: str = "company",
        timeout: float | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        request_id: str | None = None,
        cancel: "threading.Event | None" = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        role = str(role or "").strip()
        model = str(model or "").strip()
        classification = str(data_classification or "").strip().upper()
        source = str(source or "").strip().lower()
        safe_role = role if _ROLE_PATTERN.fullmatch(role) else None
        safe_model = model if model in ALLOWED_MODEL_IDS else None
        safe_classification = (
            classification if classification in _KNOWN_CLASSIFICATIONS else "UNSPECIFIED"
        )

        if classification != "PUBLIC" or source != "public_reference":
            self._emit({
                "provider": self.provider,
                "operation": "chat.completions",
                "role": safe_role,
                "model": safe_model,
                "classification": safe_classification,
                "outcome": "blocked",
                "attempts": 0,
                "duration_ms": round((time.monotonic() - started) * 1000),
            })
            raise PermanentRuntimeError("FreeLLMAPI blocks non-public or non-public-reference data")
        allowed_models = self.role_model_allowlist.get(role)
        if allowed_models is None:
            raise PermanentRuntimeError("role is not in the FreeLLMAPI allowlist")
        if model not in allowed_models:
            raise PermanentRuntimeError("model is not allowlisted for this FreeLLMAPI role")
        if not isinstance(prompt, str) or not prompt.strip():
            raise PermanentRuntimeError("prompt is required")

        request_timeout = self.timeout_seconds
        if timeout is not None:
            request_timeout = min(self.timeout_seconds, _bounded_number(timeout, "timeout", 1, 3600))
        token_limit = self.max_tokens if max_tokens is None else max_tokens
        if isinstance(token_limit, bool) or not isinstance(token_limit, int):
            raise PermanentRuntimeError("max_tokens must be an integer")
        if not 1 <= token_limit <= self.max_tokens:
            raise PermanentRuntimeError("max_tokens exceeds the FreeLLMAPI limit")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise PermanentRuntimeError("temperature must be numeric")
        if not 0 <= float(temperature) <= 2:
            raise PermanentRuntimeError("temperature must be between 0 and 2")
        if cancel is not None and cancel.is_set():
            raise FreeLLMAPICancelled("FreeLLMAPI request cancelled before dispatch")
        request_id = str(request_id or uuid.uuid4().hex)
        # Quota and the breaker are checked before any network access so an
        # exhausted or failing gateway never costs a timeout per task.
        self._guard_dispatch(request_id, safe_role, safe_model, started)

        attempts = 0
        try:
            models, discovery_attempts = self._list_models(
                min(8.0, request_timeout), cancel=cancel
            )
            attempts += discovery_attempts
            if model not in models:
                raise RuntimeUnavailable("allowlisted model is not ready in FreeLLMAPI /v1/models")
            message, response, _, request_attempts, status = self._chat_completion(
                [{"role": "user", "content": prompt}],
                model,
                request_timeout,
                token_limit,
                temperature,
                request_id=request_id,
                cancel=cancel,
            )
            attempts += request_attempts
            text = message.get("content")
            if not isinstance(text, str) or not text.strip():
                raise PermanentRuntimeError("FreeLLMAPI returned empty assistant content")
            response_model = response.get("model")
            session_id = response.get("id")
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            total_tokens = usage.get("total_tokens")
            self.quota.note_call(
                self.provider,
                int(total_tokens) if isinstance(total_tokens, (int, float)) else 0,
            )
            self.breaker.record_success(self.provider)
            self._note_latency(time.monotonic() - started)
            result = {
                "text": text,
                "provider": self.provider,
                "model": model,
                "response_model": response_model if isinstance(response_model, str) else None,
                "session_id": session_id if isinstance(session_id, str) else None,
                "usage": usage,
                "latency_seconds": round(time.monotonic() - started, 3),
                "http_status": status,
                "attempts": attempts,
                "request_id": request_id,
                "data_classification": "PUBLIC",
                "transport": "freellmapi-openai-compatible",
                "quota": self.quota.snapshot(self.provider),
            }
            self._emit({
                "provider": self.provider,
                "operation": "chat.completions",
                "role": role,
                "model": model,
                "classification": "PUBLIC",
                "outcome": "success",
                "session_id": result["session_id"],
                "request_id": request_id,
                "attempts": attempts,
                "duration_ms": round((time.monotonic() - started) * 1000),
                "http_status": status,
            })
            return result
        except (PermanentRuntimeError, RuntimeUnavailable, TransientRuntimeError) as exc:
            attempts += getattr(exc, "attempts", 0)
            if not isinstance(exc, FreeLLMAPICancelled):
                self.breaker.record_failure(self.provider, type(exc).__name__ + ": " + str(exc)[:150])
            self._emit({
                "provider": self.provider,
                "operation": "chat.completions",
                "role": role,
                "model": model,
                "classification": "PUBLIC",
                "outcome": "error",
                "error_type": type(exc).__name__,
                "request_id": request_id,
                "attempts": attempts,
                "duration_ms": round((time.monotonic() - started) * 1000),
            })
            raise

