"""Free model provider registry for VEO AI Company.

The company may only run on providers that are free for the organisation.  This
module is the single place that says *which* providers are actually free, what
they cost in quota, and whether they may receive company data at all.

Three free classes exist and they are never mixed up:

``LOCAL``
    ``ollama`` and ``docker-model-runner``.  Free because the hardware is ours.
    They may hold PRIVATE company work and they are the worker's primary route.

``HOSTED_FREE``
    Recurring free API tiers (Groq, Gemini, Cloudflare Workers AI, OpenRouter
    ``:free``).  Free because the vendor publishes a standing free allowance.
    They may only ever receive ``PUBLIC`` data, so they are never eligible for
    a PRIVATE company task.

``INELIGIBLE``
    One-time trial credit (Cerebras), retired services (GitHub Models), tiers
    with no included hosted inference (Hugging Face free accounts) and vendors
    whose free plan has no API access (Mistral).  Reported for completeness,
    never routable, never described as "free".

Statuses reported to operators are strictly these five:

``LIVE_VERIFIED``            real health + model list + real inference succeeded
``CONFIGURED_NOT_VERIFIED``  credentials present, no live call made yet
``UNIT_TEST_ONLY``           exercised against a stub endpoint only
``DISABLED``                 not a recurring free tier, or switched off in config
``BLOCKED``                  free provider that cannot be called (no credential,
                             no listener, account id missing)

Nothing in this module can reach a paid endpoint: every spec carries a
``base_url`` and there is no code path that accepts an arbitrary URL.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jcode_runtime import PermanentRuntimeError, RuntimeUnavailable, TransientRuntimeError

PUBLIC = "PUBLIC"
PRIVATE = "PRIVATE"
PROBE_MARKER = "VEO_FREE_MODEL_OK"

LIVE_VERIFIED = "LIVE_VERIFIED"
CONFIGURED_NOT_VERIFIED = "CONFIGURED_NOT_VERIFIED"
UNIT_TEST_ONLY = "UNIT_TEST_ONLY"
DISABLED = "DISABLED"
BLOCKED = "BLOCKED"

# Classes
LOCAL = "LOCAL"
HOSTED_FREE = "HOSTED_FREE"

# Circuit breaker states, reported verbatim so operators stop guessing.
BREAKER_CLOSED = "CLOSED"
BREAKER_OPEN = "OPEN"
BREAKER_HALF_OPEN = "HALF_OPEN"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class FreeProviderSpec:
    """One provider, with the facts that decide whether it may be used.

    Quota fields are ``None`` where the vendor publishes no fixed number; the
    rate-limit path (HTTP 429 handling) still applies in that case.
    """

    name: str
    kind: str
    billing_class: str
    base_url: str | None
    api_key_env: str
    models: tuple[str, ...]
    context_tokens: int
    requests_per_minute: int | None
    requests_per_day: int | None
    tokens_per_day: int | None
    tools: bool
    trains_on_free_inputs: bool
    recurring_free: bool
    docs_url: str
    notes: str
    account_id_env: str | None = None
    free_model_suffix: str | None = None
    ineligible_reason: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "kind": self.kind,
            "billing_class": self.billing_class,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "documented_models": list(self.models),
            "context_tokens": self.context_tokens,
            "requests_per_minute": self.requests_per_minute,
            "requests_per_day": self.requests_per_day,
            "tokens_per_day": self.tokens_per_day,
            "tool_calling": self.tools,
            "trains_on_free_inputs": self.trains_on_free_inputs,
            "recurring_free": self.recurring_free,
            "docs_url": self.docs_url,
            "notes": self.notes,
            "account_id_env": self.account_id_env,
            "ineligible_reason": self.ineligible_reason,
        }


def _spec(**kwargs) -> FreeProviderSpec:
    defaults = dict(
        account_id_env=None,
        free_model_suffix=None,
        ineligible_reason=None,
        tools=True,
        trains_on_free_inputs=False,
        recurring_free=True,
    )
    defaults.update(kwargs)
    return FreeProviderSpec(**defaults)


# Verified 2026-11-23 against the vendors' published documentation.  Numeric
# quotas are the published free-plan figures; they change, so the registry
# table is evidence with a date, not a promise.
PROVIDER_SPECS: tuple[FreeProviderSpec, ...] = (
    _spec(
        name="groq",
        kind="hosted_free",
        billing_class=HOSTED_FREE,
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        models=("openai/gpt-oss-20b", "openai/gpt-oss-120b"),
        context_tokens=131072,
        requests_per_minute=30,
        requests_per_day=1000,
        tokens_per_day=200000,
        tools=True,
        trains_on_free_inputs=False,
        recurring_free=True,
        docs_url="https://console.groq.com/docs/rate-limits",
        notes="Standing free plan, organisation-level limits, 8000 tokens/minute.",
    ),
    _spec(
        name="gemini",
        kind="hosted_free",
        billing_class=HOSTED_FREE,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        models=("gemini-3.8-flash",),
        context_tokens=1048576,
        requests_per_minute=None,
        requests_per_day=None,
        tokens_per_day=None,
        tools=True,
        # Google may use unpaid-service inputs/outputs to improve products.
        trains_on_free_inputs=True,
        recurring_free=True,
        docs_url="https://ai.google.dev/gemini-api/docs/rate-limits",
        notes="Free tier is real but project-specific and unpaid data may be used by Google.",
    ),
    _spec(
        name="cloudflare",
        kind="hosted_free",
        billing_class=HOSTED_FREE,
        base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        api_key_env="CLOUDFLARE_API_TOKEN",
        models=("@cf/zai-org/glm-4.7-flash",),
        context_tokens=131072,
        requests_per_minute=300,
        requests_per_day=None,
        tokens_per_day=None,
        tools=True,
        trains_on_free_inputs=False,
        recurring_free=True,
        docs_url="https://developers.cloudflare.com/workers-ai/platform/pricing",
        notes="10000 Neurons/day shared per account; some models need the paid plan.",
        account_id_env="CLOUDFLARE_ACCOUNT_ID",
    ),
    _spec(
        name="openrouter",
        kind="hosted_free",
        billing_class=HOSTED_FREE,
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        models=("meta-llama/llama-3.2-3b-instruct:free",),
        context_tokens=131072,
        requests_per_minute=20,
        requests_per_day=50,
        tokens_per_day=None,
        tools=True,
        trains_on_free_inputs=False,
        recurring_free=True,
        docs_url="https://openrouter.ai/docs/api_reference/limits",
        notes="Only ':free' model variants are callable at no charge; 50 requests/day.",
        free_model_suffix=":free",
    ),
    _spec(
        name="cerebras",
        kind="trial_credit",
        billing_class="TRIAL",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        models=("gpt-oss-120b",),
        context_tokens=65536,
        requests_per_minute=5,
        requests_per_day=None,
        tokens_per_day=1000000,
        tools=True,
        trains_on_free_inputs=False,
        recurring_free=False,
        docs_url="https://inference-docs.cerebras.ai/support/rate-limits",
        notes="5 USD one-off trial credit expiring after 30 days; not recurring free.",
        ineligible_reason="TRIAL_CREDIT_NOT_RECURRING_FREE",
    ),
    _spec(
        name="huggingface",
        kind="no_free_tier",
        billing_class="PAID",
        base_url="https://router.huggingface.co/v1",
        api_key_env="HF_TOKEN",
        models=(),
        context_tokens=0,
        requests_per_minute=None,
        requests_per_day=None,
        tokens_per_day=None,
        tools=True,
        trains_on_free_inputs=False,
        recurring_free=False,
        docs_url="https://huggingface.co/docs/inference-providers/pricing",
        notes="Free accounts have no included inference credit; PRO credits are paid.",
        ineligible_reason="NO_INCLUDED_FREE_CREDIT",
    ),
    _spec(
        name="github-models",
        kind="retired",
        billing_class="RETIRED",
        base_url=None,
        api_key_env="GITHUB_MODELS_TOKEN",
        models=(),
        context_tokens=0,
        requests_per_minute=None,
        requests_per_day=None,
        tokens_per_day=None,
        tools=False,
        trains_on_free_inputs=False,
        recurring_free=False,
        docs_url="https://docs.github.com/en/github-models",
        notes="GitHub Models was retired on 2026-07-30.",
        ineligible_reason="SERVICE_RETIRED",
    ),
    _spec(
        name="mistral",
        kind="no_free_api",
        billing_class="PAID",
        base_url="https://api.mistral.ai/openai/v1",
        api_key_env="MISTRAL_API_KEY",
        models=(),
        context_tokens=0,
        requests_per_minute=None,
        requests_per_day=None,
        tokens_per_day=None,
        tools=True,
        trains_on_free_inputs=None,
        recurring_free=False,
        docs_url="https://docs.mistral.ai/admin/billing-usage/usage-limits",
        notes="Free plan does not include API access; API use requires billing.",
        ineligible_reason="FREE_PLAN_HAS_NO_API_ACCESS",
    ),
)

# Free local engines.  Their capability cards come from live discovery, not
# from this table; the entries exist so one registry answers "what is free".
LOCAL_ENGINES: tuple[dict[str, str], ...] = (
    {
        "provider": "ollama",
        "billing_class": LOCAL,
        "endpoint": "http://127.0.0.1:11434",
        "reason_free": "self-hosted open weights, no per-token fee",
        "classification_allowance": "PRIVATE and PUBLIC",
    },
    {
        "provider": "docker-model-runner",
        "billing_class": LOCAL,
        "endpoint": "http://127.0.0.1:5005/engines/v1",
        "reason_free": "self-hosted open weights, no per-token fee",
        "classification_allowance": "PRIVATE and PUBLIC",
    },
    {
        "provider": "freellmapi",
        "billing_class": "LOCAL_GATEWAY",
        "endpoint": "http://127.0.0.1:3001/v1",
        "reason_free": "loopback gateway, disabled by default config",
        "classification_allowance": "PUBLIC only",
    },
)


def spec_by_name(name: str) -> FreeProviderSpec | None:
    for spec in PROVIDER_SPECS:
        if spec.name == name:
            return spec
    return None


# --------------------------------------------------------------------- quota
class QuotaLedger:
    """Per-provider request/token counters so quota exhaustion is visible.

    The ledger is deliberately simple: process-local counters, optionally
    persisted as JSON.  It cannot know what the vendor counts, so it is used to
    stop the company hammering a free tier, never to claim remaining quota.
    """

    # Counters that only ever grow inside one day/minute bucket.
    _ADDITIVE = ("calls_today", "tokens_today", "calls_this_minute", "rate_limited")

    def __init__(self, path: str | Path | None = None, lock_timeout: float = 10.0) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self.lock_timeout = max(0.5, float(lock_timeout))
        self._state: dict[str, dict[str, Any]] = {}
        # What this writer last persisted, so a flush adds the delta instead of
        # overwriting another worker's counters.
        self._written: dict[str, dict[str, Any]] = {}
        if self.path is not None:
            self._load()

    # -- persistence ------------------------------------------------------
    @staticmethod
    def _read_payload(path: Path) -> dict[str, dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(provider): dict(row)
            for provider, row in payload.items()
            if isinstance(row, dict)
        }

    def _load(self) -> None:
        with self._lock:
            self._state = self._read_payload(self.path)
            self._written = {name: dict(row) for name, row in self._state.items()}

    def _lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    def _acquire_file_lock(self):
        """Cross-process exclusive lock so two workers never clobber the file.

        A lock left behind by a crashed process would stall the worker forever,
        so the wait is bounded and the stale lock is broken rather than skipped.
        """
        if self.path is None:
            return None
        lock_path = self._lock_path()
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                handle = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(handle)
                return lock_path
            except FileExistsError:
                if time.monotonic() >= deadline:
                    try:
                        lock_path.unlink()
                    except OSError:
                        pass
                    continue
                time.sleep(0.02)
            except OSError:
                return None

    @staticmethod
    def _release_file_lock(lock_path) -> None:
        if lock_path is None:
            return
        try:
            lock_path.unlink()
        except OSError:
            pass

    def _flush(self) -> None:
        """Persist one delta atomically, merging with what is already on disk."""
        if self.path is None:
            return
        lock_path = self._acquire_file_lock()
        try:
            on_disk = self._read_payload(self.path)
            merged: dict[str, dict[str, Any]] = {}
            for provider, row in self._state.items():
                stored = dict(on_disk.get(provider) or {})
                if stored.get("day") != row.get("day") or stored.get("minute") != row.get("minute"):
                    # The bucket rolled over since anyone last wrote it.
                    stored = {"day": row.get("day"), "minute": row.get("minute")}
                previous = self._written.get(provider) or {}
                same_bucket = (
                    previous.get("day") == row.get("day")
                    and previous.get("minute") == row.get("minute")
                )
                merged_row = dict(stored)
                for key in self._ADDITIVE:
                    mine = int(row.get(key) or 0)
                    before = int(previous.get(key) or 0) if same_bucket else 0
                    merged_row[key] = int(stored.get(key) or 0) + max(0, mine - before)
                merged_row["day"] = row.get("day")
                merged_row["minute"] = row.get("minute")
                if row.get("last_error"):
                    merged_row["last_error"] = row["last_error"]
                merged[provider] = merged_row
            # Rows written by other workers but untouched here must survive.
            for provider, row in on_disk.items():
                merged.setdefault(provider, row)
            self._write_atomic(merged)
            # Track what *this* writer contributed, never the merged total, or the
            # next delta would go negative and silently drop real calls.
            self._written = {name: dict(row) for name, row in self._state.items()}
        finally:
            self._release_file_lock(lock_path)

    def _write_atomic(self, payload: dict[str, dict[str, Any]]) -> None:
        temp = self.path.with_name(self.path.name + ".tmp" + str(os.getpid()))
        try:
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(str(temp), str(self.path))
        except OSError:
            try:
                temp.unlink()
            except OSError:
                pass

    @staticmethod
    def _minute() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")

    @staticmethod
    def _day() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _row(self, provider: str) -> dict[str, Any]:
        row = self._state.setdefault(provider, {})
        if row.get("day") != self._day():
            row["day"] = self._day()
            row["calls_today"] = 0
            row["tokens_today"] = 0
            row["rate_limited"] = 0
        if row.get("minute") != self._minute():
            row["minute"] = self._minute()
            row["calls_this_minute"] = 0
        row.setdefault("calls_today", 0)
        row.setdefault("tokens_today", 0)
        row.setdefault("calls_this_minute", 0)
        row.setdefault("rate_limited", 0)
        row.setdefault("last_error", None)
        return row

    # -- api --------------------------------------------------------------
    def snapshot(self, provider: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._state.get(provider) or {})

    def can_call(self, spec: FreeProviderSpec) -> tuple[bool, str]:
        with self._lock:
            row = self._row(spec.name)
            if spec.requests_per_minute and row["calls_this_minute"] >= spec.requests_per_minute:
                return False, "FREE_QUOTA_RPM_EXHAUSTED"
            if spec.requests_per_day and row["calls_today"] >= spec.requests_per_day:
                return False, "FREE_QUOTA_RPD_EXHAUSTED"
            if spec.tokens_per_day and row["tokens_today"] >= spec.tokens_per_day:
                return False, "FREE_QUOTA_TOKENS_EXHAUSTED"
            return True, "ok"

    def note_call(self, provider: str, tokens: int = 0, requests: int = 1) -> None:
        request_count = max(1, int(requests))
        with self._lock:
            row = self._row(provider)
            row["calls_today"] += request_count
            row["calls_this_minute"] += request_count
            if tokens > 0:
                row["tokens_today"] += int(tokens)
            self._flush()

    def note_rate_limited(self, provider: str, detail: str = "") -> None:
        with self._lock:
            row = self._row(provider)
            row["rate_limited"] += 1
            row["last_error"] = ("429 " + detail)[:200]
            self._flush()

    def note_error(self, provider: str, detail: str) -> None:
        with self._lock:
            row = self._row(provider)
            row["last_error"] = str(detail)[:200]
            self._flush()

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return {name: dict(row) for name, row in sorted(self._state.items())}


# ------------------------------------------------------------------ breaker
class CircuitBreaker:
    """Bounded per-provider failure isolation.

    Three consecutive failures open the circuit for a cooldown; after that a
    single probe is allowed through.  This keeps a dead free tier from being
    retried on every task while still recovering on its own.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be positive")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be positive")
        self.failure_threshold = int(failure_threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self._lock = threading.Lock()
        self._state: dict[str, dict[str, Any]] = {}

    def _row(self, provider: str) -> dict[str, Any]:
        return self._state.setdefault(
            provider,
            {
                "state": BREAKER_CLOSED,
                "failures": 0,
                "opened_at": None,
                "half_open_inflight": False,
                "last_error": None,
                "successes": 0,
            },
        )

    def allow(self, provider: str) -> bool:
        """Admit one call: CLOSED admits all, OPEN none, HALF_OPEN exactly one."""
        with self._lock:
            row = self._row(provider)
            state = row.get("state") or BREAKER_CLOSED
            if state == BREAKER_CLOSED:
                return True
            if state == BREAKER_OPEN:
                opened = row.get("opened_at")
                if opened is None or time.monotonic() - float(opened) < self.cooldown_seconds:
                    return False
                row["state"] = BREAKER_HALF_OPEN
            # HALF_OPEN: a single probe is in flight at a time, so a recovering
            # provider is not hit by every worker at once.
            if row.get("half_open_inflight"):
                return False
            row["half_open_inflight"] = True
            return True

    def state_of(self, provider: str) -> str:
        with self._lock:
            row = self._row(provider)
            state = str(row.get("state") or BREAKER_CLOSED)
            opened = row.get("opened_at")
            if (
                state == BREAKER_OPEN
                and opened is not None
                and time.monotonic() - float(opened) >= self.cooldown_seconds
            ):
                return BREAKER_HALF_OPEN
            return state

    def record_success(self, provider: str) -> None:
        with self._lock:
            row = self._row(provider)
            row["failures"] = 0
            row["opened_at"] = None
            row["state"] = BREAKER_CLOSED
            row["half_open_inflight"] = False
            row["successes"] = row.get("successes", 0) + 1
            row["last_error"] = None

    def record_failure(self, provider: str, detail: str = "") -> None:
        with self._lock:
            row = self._row(provider)
            row["half_open_inflight"] = False
            row["last_error"] = str(detail)[:200]
            if row.get("state") == BREAKER_HALF_OPEN:
                # A failed half-open probe re-opens immediately, whatever the
                # failure count was, so recovery never flaps.
                row["failures"] = max(int(row.get("failures") or 0), self.failure_threshold)
                row["state"] = BREAKER_OPEN
                row["opened_at"] = time.monotonic()
                return
            row["failures"] = row.get("failures", 0) + 1
            if row["failures"] >= self.failure_threshold:
                row["state"] = BREAKER_OPEN
                row["opened_at"] = time.monotonic()

    def describe(self) -> dict[str, Any]:
        with self._lock:
            result = {}
            for provider, row in sorted(self._state.items()):
                state = str(row.get("state") or BREAKER_CLOSED)
                opened = row.get("opened_at")
                retry_in = 0.0
                if state == BREAKER_OPEN and opened is not None:
                    remaining = self.cooldown_seconds - (time.monotonic() - float(opened))
                    retry_in = round(max(0.0, remaining), 1)
                result[provider] = {
                    "state": state,
                    "failures": row.get("failures", 0),
                    "successes": row.get("successes", 0),
                    "open": state == BREAKER_OPEN,
                    "half_open_inflight": bool(row.get("half_open_inflight")),
                    "retry_in_seconds": retry_in,
                    "last_error": row.get("last_error"),
                }
            return result


# ----------------------------------------------------------------- provider
class HostedFreeProvider:
    """OpenAI-compatible client for a recurring-free hosted tier.

    Contract with the router: ``execute`` refuses anything that is not
    ``PUBLIC`` *before* opening a socket, so a PRIVATE company task can never
    leak to a vendor by mistake.
    """

    def __init__(
        self,
        spec: FreeProviderSpec,
        api_key: str | None = None,
        base_url: str | None = None,
        account_id: str | None = None,
        quota: QuotaLedger | None = None,
        breaker: CircuitBreaker | None = None,
        timeout: float = 45,
        max_tokens: int = 512,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        retry_after_cap: float = 10.0,
    ) -> None:
        if not spec.recurring_free:
            raise RuntimeUnavailable(
                spec.name + " is not a recurring free tier (" + str(spec.ineligible_reason) + ")"
            )
        if spec.base_url is None:
            raise RuntimeUnavailable(spec.name + " has no usable endpoint")
        resolved = base_url or spec.base_url
        if spec.account_id_env and "{account_id}" in resolved:
            if not account_id:
                raise RuntimeUnavailable(spec.name + " requires " + spec.account_id_env)
            resolved = resolved.replace("{account_id}", account_id)
        self.spec = spec
        self.provider = spec.name
        self.class_ = HOSTED_FREE
        self.paid = False
        self.base_url = resolved.rstrip("/")
        self.api_key = api_key
        self.quota = quota or QuotaLedger()
        self.breaker = breaker or CircuitBreaker()
        self.timeout = max(5.0, float(timeout))
        self.max_tokens = max(16, int(max_tokens))
        self.max_retries = max(0, min(2, int(max_retries)))
        self.retry_delay = max(0.0, float(retry_delay))
        # A vendor asking us to wait longer than this is not retried in-process:
        # the caller sees a transient error with retry_after instead of blocking.
        self.retry_after_cap = max(0.0, float(retry_after_cap))

    # -- identity ---------------------------------------------------------
    def descriptor(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "class": HOSTED_FREE,
            "paid": False,
            "base_url": self.base_url,
            "documented_models": list(self.spec.models),
            "context_tokens": self.spec.context_tokens,
            "tool_calling": self.spec.tools,
            "trains_on_free_inputs": self.spec.trains_on_free_inputs,
            "allowed_classifications": [PUBLIC],
            "api_key_source": "environment",
            "api_key_env": self.spec.api_key_env,
            "quota": {
                "requests_per_minute": self.spec.requests_per_minute,
                "requests_per_day": self.spec.requests_per_day,
                "tokens_per_day": self.spec.tokens_per_day,
            },
            "tool_profile": "none",
        }

    # -- transport --------------------------------------------------------
    @staticmethod
    def _open(request: urllib.request.Request, timeout: float):
        # Ignore ambient proxies: these are direct vendor calls with an API key.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=timeout)

    @staticmethod
    def _retry_after(exc: "urllib.error.HTTPError") -> float | None:
        """Parse ``Retry-After`` (delta-seconds or HTTP-date); ``None`` if absent."""
        raw = None
        try:
            raw = exc.headers.get("Retry-After") if exc.headers else None
        except Exception:
            return None
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return max(0.0, float(text))
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())

    @staticmethod
    def _transient(message: str, retry_after: float | None) -> TransientRuntimeError:
        error = TransientRuntimeError(message)
        try:
            error.retry_after = retry_after  # type: ignore[attr-defined]
        except Exception:
            pass
        return error

    def _authorised(self) -> str:
        key = (self.api_key or "").strip()
        if not key:
            raise RuntimeUnavailable(
                self.provider + " needs a free-tier key in " + self.spec.api_key_env
            )
        if "\r" in key or "\n" in key:
            raise PermanentRuntimeError(self.provider + " API key value is invalid")
        return key

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> tuple[dict[str, Any], int, int]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json", "Authorization": "Bearer " + self._authorised()}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                with self._open(request, timeout) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    status = int(getattr(response, "status", 200))
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                message = self.provider + " HTTP " + str(exc.code) + ": " + detail
                retry_after = self._retry_after(exc)
                retryable = exc.code in (408, 425, 429) or 500 <= int(exc.code) < 600
                if exc.code == 429:
                    self.quota.note_rate_limited(
                        self.provider,
                        detail + (" ; retry-after=" + str(retry_after) if retry_after else ""),
                    )
                within_cap = retry_after is None or retry_after <= self.retry_after_cap
                if retryable and attempt <= self.max_retries and within_cap:
                    time.sleep(retry_after if retry_after is not None else self.retry_delay * attempt)
                    last = TransientRuntimeError(message)
                    continue
                if retryable:
                    self.breaker.record_failure(self.provider, message)
                    raise self._transient(message, retry_after) from None
                self.breaker.record_failure(self.provider, message)
                raise PermanentRuntimeError(message) from None
            except (socket.timeout, TimeoutError) as exc:
                # Retrying a timeout doubles the wall clock and can outlive the
                # task lease, so it fails bounded and immediately.
                message = (
                    self.provider + " timed out after " + str(timeout) + "s: "
                    + type(exc).__name__
                )
                self.breaker.record_failure(self.provider, message)
                raise TransientRuntimeError(message) from None
            except Exception as exc:  # transport failure
                message = self.provider + " transport failed: " + type(exc).__name__ + ": " + str(exc)[:200]
                if attempt <= self.max_retries:
                    time.sleep(self.retry_delay * attempt)
                    last = TransientRuntimeError(message)
                    continue
                self.breaker.record_failure(self.provider, message)
                raise TransientRuntimeError(message) from None
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                self.breaker.record_failure(self.provider, "invalid JSON")
                raise PermanentRuntimeError(self.provider + " returned invalid JSON") from None
            if not isinstance(parsed, dict):
                self.breaker.record_failure(self.provider, "non-object response")
                raise PermanentRuntimeError(self.provider + " returned a non-object response") from None
            return parsed, attempt, status
        raise last or RuntimeUnavailable(self.provider + " request failed")

    # -- discovery --------------------------------------------------------
    def list_models(self, timeout: float = 8) -> list[str]:
        payload, _, _ = self._request("GET", "/models", None, timeout)
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise RuntimeUnavailable(self.provider + " /models response has no data list")
        names = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            model = row.get("id")
            if not isinstance(model, str) or not model:
                continue
            if self.spec.free_model_suffix and not model.endswith(self.spec.free_model_suffix):
                # A paid variant must never become a route.
                continue
            names.append(model)
        if not names:
            raise RuntimeUnavailable(
                self.provider + " reported no free models at " + now_iso()
            )
        return sorted(set(names))

    def health(self, timeout: float = 8) -> dict[str, Any]:
        started = time.monotonic()
        result: dict[str, Any] = {
            "status": "error",
            "latency_seconds": None,
            "models": [],
            "models_count": 0,
            "checked_at": now_iso(),
            "error": None,
            "circuit_open": not self.breaker.allow(self.provider),
        }
        if result["circuit_open"]:
            result["error"] = "CIRCUIT_OPEN"
            return result
        try:
            result["models"] = self.list_models(timeout=timeout)
            result["models_count"] = len(result["models"])
            result["status"] = "ok"
            self.breaker.record_success(self.provider)
        except Exception as exc:
            result["error"] = type(exc).__name__ + ": " + str(exc)[:200]
        result["latency_seconds"] = round(time.monotonic() - started, 2)
        return result

    def reachable(self, timeout: float = 5) -> bool:
        return self.health(timeout=timeout)["status"] == "ok"

    # -- execution --------------------------------------------------------
    def _model_for(self, model: str | None, discovered: list[str]) -> str:
        wanted = str(model or "").strip()
        if wanted:
            if self.spec.free_model_suffix and not wanted.endswith(self.spec.free_model_suffix):
                raise PermanentRuntimeError(
                    self.provider + " refuses the non-free model variant: " + wanted
                )
            return wanted
        if discovered:
            return discovered[0]
        if self.spec.models:
            return self.spec.models[0]
        raise RuntimeUnavailable(self.provider + " has no model to call")

    def execute(
        self,
        prompt: str,
        workspace: Path | str,
        model: str | None = None,
        timeout: float | None = None,
        tool_profile: str | None = None,
        max_tokens: int | None = None,
        data_classification: str = PRIVATE,
    ) -> dict[str, Any]:
        classification = str(data_classification or "").strip().upper()
        if classification != PUBLIC:
            # Fail closed before any credential or network access.
            raise PermanentRuntimeError(
                self.provider + " may receive PUBLIC data only; task is "
                + (classification or "UNSPECIFIED")
            )
        prompt = str(prompt or "").strip()
        if not prompt:
            raise PermanentRuntimeError("prompt is required")
        Path(workspace).mkdir(parents=True, exist_ok=True)
        can_call, reason = self.quota.can_call(self.spec)
        if not can_call:
            raise RuntimeUnavailable(self.provider + " free quota exhausted: " + reason)
        if not self.breaker.allow(self.provider):
            raise TransientRuntimeError(self.provider + " circuit is open after repeated failures")

        discovered: list[str] = []
        try:
            discovered = self.list_models(timeout=min(8.0, float(timeout or self.timeout)))
        except PermanentRuntimeError:
            # A rejected credential must not be followed by a paid-path guess.
            raise
        except Exception:
            # A transient discovery failure is not fatal on its own: the chat
            # call below reports the real error.  Quota is still tracked.
            pass
        selected = self._model_for(model, discovered)
        limit = self.max_tokens if max_tokens is None else int(max_tokens)
        if limit < 1 or limit > self.max_tokens:
            raise PermanentRuntimeError("max_tokens exceeds the " + self.provider + " limit")
        payload = {
            "model": selected,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.2,
            "max_tokens": limit,
        }
        started = time.monotonic()
        body, attempts, status = self._request(
            "POST", "/chat/completions", payload, float(timeout or self.timeout)
        )
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            self.breaker.record_failure(self.provider, "no completion choice")
            raise PermanentRuntimeError(self.provider + " returned no completion choice")
        message = choices[0].get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            self.breaker.record_failure(self.provider, "empty assistant content")
            raise PermanentRuntimeError(self.provider + " returned empty assistant content")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        tokens = usage.get("total_tokens")
        self.quota.note_call(self.provider, int(tokens) if isinstance(tokens, (int, float)) else 0)
        self.breaker.record_success(self.provider)
        return {
            "text": text,
            "provider": self.provider,
            "class": HOSTED_FREE,
            "paid": False,
            "model": selected,
            "response_model": body.get("model") if isinstance(body.get("model"), str) else None,
            "tool_profile": "none",
            "session_id": body.get("id") if isinstance(body.get("id"), str) else None,
            "usage": usage,
            "latency_seconds": round(time.monotonic() - started, 2),
            "http_status": status,
            "attempts": attempts,
            "data_classification": PUBLIC,
            "transport": "openai-compatible-hosted-free",
            "quota": self.quota.snapshot(self.provider),
        }

    # -- probe ------------------------------------------------------------
    def probe(self, model: str | None = None, workspace: Path | str = ".", timeout: float | None = None) -> dict[str, Any]:
        """Real inference smoke prompt; a key with no quota shows up as failure."""
        started = time.monotonic()
        result: dict[str, Any] = {
            "provider": self.provider,
            "class": HOSTED_FREE,
            "paid": False,
            "model": model,
            "models": [],
            "available": False,
            "ok": False,
            "marker_ok": False,
            "reason": "",
            "latency_seconds": None,
            "base_url": self.base_url,
            "checked_at": now_iso(),
            "smoke_text": "",
            "usage": {},
        }
        try:
            models = self.list_models(timeout=min(8.0, float(timeout or self.timeout)))
            result["models"] = models
            selected = self._model_for(model, models)
            result["model"] = selected
            smoke = self.execute(
                "Reply with exactly " + PROBE_MARKER + " and nothing else.",
                workspace=workspace,
                model=selected,
                timeout=timeout,
                max_tokens=24,
                data_classification=PUBLIC,
            )
            result["smoke_text"] = smoke["text"][:300]
            result["usage"] = smoke.get("usage") or {}
            result["latency_seconds"] = smoke.get("latency_seconds")
            result["marker_ok"] = PROBE_MARKER in smoke["text"]
            result["available"] = True
            result["ok"] = result["marker_ok"]
            result["reason"] = (
                self.provider + " answered with " + PROBE_MARKER
                if result["marker_ok"]
                else self.provider + " answered without the readiness marker"
            )
        except Exception as exc:
            result["reason"] = type(exc).__name__ + ": " + str(exc)[:300]
        if result["latency_seconds"] is None:
            result["latency_seconds"] = round(time.monotonic() - started, 2)
        return result


# ---------------------------------------------------------------- registry
ENTRY_FIELDS = (
    "provider",
    "status",
    "kind",
    "billing_class",
    "recurring_free",
    "eligible_for_tasks",
    "ineligible_reason",
    "credential_present",
    "api_key_env",
    "base_url",
    "classification_allowance",
    "documented_models",
    "discovered_models",
    "quota",
    "health",
    "checked_at",
    "notes",
)


class FreeProviderRegistry:
    """Decide, from the live environment, which free providers may be used."""

    def __init__(
        self,
        specs: tuple[FreeProviderSpec, ...] | None = None,
        env: dict[str, str] | None = None,
        quota: QuotaLedger | None = None,
        breaker: CircuitBreaker | None = None,
        workspace: Path | str | None = None,
        max_tokens: int = 512,
        timeout: float = 45,
        overrides: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.specs = tuple(specs or PROVIDER_SPECS)
        # Explicit per-provider overrides (used by tests and local stubs).  An
        # override can supply a credential or a stub base URL; it can never
        # point a provider at a different vendor class.
        self.overrides = {key: dict(value) for key, value in (overrides or {}).items()}
        self.env = dict(os.environ if env is None else env)
        self.quota = quota or QuotaLedger()
        self.breaker = breaker or CircuitBreaker()
        self.workspace = Path(workspace or ".")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.entries: dict[str, dict[str, Any]] = {}
        self.providers: dict[str, HostedFreeProvider] = {}
        for spec in self.specs:
            self._register(spec)

    def _register(self, spec: FreeProviderSpec) -> None:
        entry: dict[str, Any] = {
            "provider": spec.name,
            "status": DISABLED,
            "kind": spec.kind,
            "billing_class": spec.billing_class,
            "recurring_free": spec.recurring_free,
            "eligible_for_tasks": False,
            "ineligible_reason": spec.ineligible_reason,
            "credential_present": bool(self.env.get(spec.api_key_env)),
            "api_key_env": spec.api_key_env,
            "base_url": spec.base_url,
            "classification_allowance": PUBLIC,
            "documented_models": list(spec.models),
            "discovered_models": [],
            "quota": {
                "requests_per_minute": spec.requests_per_minute,
                "requests_per_day": spec.requests_per_day,
                "tokens_per_day": spec.tokens_per_day,
            },
            "health": None,
            "checked_at": now_iso(),
            "notes": spec.notes,
        }
        self.entries[spec.name] = entry
        if not spec.recurring_free:
            entry["status"] = DISABLED
            entry["ineligible_reason"] = spec.ineligible_reason or "NOT_RECURRING_FREE"
            return
        override = self.overrides.get(spec.name, {})
        api_key = override.get("api_key") or self.env.get(spec.api_key_env)
        if override.get("api_key"):
            entry["credential_present"] = True
            entry["api_key_source"] = "override"
        if not api_key:
            entry["status"] = BLOCKED
            entry["ineligible_reason"] = "MISSING_FREE_TIER_CREDENTIAL"
            entry["notes"] = spec.notes + " Set " + spec.api_key_env + " to enable."
            return
        account_id = (
            override.get("account_id")
            or (self.env.get(spec.account_id_env) if spec.account_id_env else None)
        )
        if spec.account_id_env and not account_id:
            entry["status"] = BLOCKED
            entry["ineligible_reason"] = "MISSING_ACCOUNT_ID"
            return
        try:
            provider = HostedFreeProvider(
                spec,
                api_key=api_key,
                base_url=override.get("base_url"),
                account_id=account_id,
                quota=self.quota,
                breaker=self.breaker,
                timeout=self.timeout,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            entry["status"] = BLOCKED
            entry["ineligible_reason"] = type(exc).__name__ + ": " + str(exc)[:200]
            return
        self.providers[spec.name] = provider
        entry["status"] = CONFIGURED_NOT_VERIFIED

    # -- live probing -----------------------------------------------------
    def probe(self, inference: bool = True, timeout: float | None = None) -> dict[str, Any]:
        """Call each credentialed provider for real and record the outcome."""
        for name, provider in sorted(self.providers.items()):
            entry = self.entries[name]
            if not self.breaker.allow(name):
                entry["health"] = {"status": "error", "error": "CIRCUIT_OPEN"}
                entry["status"] = BLOCKED
                entry["ineligible_reason"] = "CIRCUIT_OPEN"
                continue
            health = provider.health(timeout=min(8.0, timeout or self.timeout))
            entry["health"] = health
            entry["checked_at"] = now_iso()
            if health["status"] != "ok":
                entry["status"] = BLOCKED
                entry["ineligible_reason"] = "HEALTH_CHECK_FAILED"
                continue
            entry["discovered_models"] = list(health.get("models") or [])
            entry["status"] = CONFIGURED_NOT_VERIFIED
            if not inference:
                continue
            probe = provider.probe(
                (health.get("models") or [None])[0],
                workspace=self.workspace,
                timeout=timeout,
            )
            entry["health"] = {**health, "inference": probe}
            entry["checked_at"] = now_iso()
            if probe.get("ok"):
                entry["status"] = LIVE_VERIFIED
                entry["eligible_for_tasks"] = True
                entry["ineligible_reason"] = None
            else:
                entry["status"] = BLOCKED
                entry["ineligible_reason"] = "INFERENCE_PROBE_FAILED"
        return self.describe()

    def describe(self) -> dict[str, Any]:
        entries = []
        for spec in self.specs:
            row = self.entries[spec.name]
            entries.append({key: row.get(key) for key in ENTRY_FIELDS})
        return {
            "checked_at": now_iso(),
            "classification_policy": "hosted free providers receive PUBLIC data only",
            "providers": entries,
            "local_engines": [dict(item) for item in LOCAL_ENGINES],
            "quota": self.quota.describe(),
            "circuit_breakers": self.breaker.describe(),
            "counts": {
                "live_verified": sum(1 for row in entries if row["status"] == LIVE_VERIFIED),
                "configured_not_verified": sum(
                    1 for row in entries if row["status"] == CONFIGURED_NOT_VERIFIED
                ),
                "blocked": sum(1 for row in entries if row["status"] == BLOCKED),
                "disabled": sum(1 for row in entries if row["status"] == DISABLED),
            },
        }

    def routable(self) -> dict[str, HostedFreeProvider]:
        """Providers that passed a live inference probe and may take PUBLIC work."""
        return {
            name: provider
            for name, provider in self.providers.items()
            if self.entries[name]["status"] == LIVE_VERIFIED
        }


def registry_from_env(
    workspace: Path | str | None = None,
    quota_path: str | Path | None = None,
    max_tokens: int = 512,
    timeout: float = 45,
) -> FreeProviderRegistry:
    """Build the registry; with no credentials this makes no network calls."""
    return FreeProviderRegistry(
        env=dict(os.environ),
        quota=QuotaLedger(quota_path),
        workspace=workspace,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="VEO free model provider registry")
    parser.add_argument("--probe", action="store_true", help="call every credentialed provider for real")
    parser.add_argument("--out", default=None, help="write the JSON registry to this path")
    args = parser.parse_args(argv)

    registry = registry_from_env(workspace=Path(__file__).resolve().parent)
    payload = registry.probe(inference=args.probe) if args.probe else registry.describe()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
