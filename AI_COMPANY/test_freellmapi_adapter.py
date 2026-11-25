import json
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import free_providers
from free_providers import QuotaLedger
from freellmapi_adapter import (
    DEFAULT_BASE_URL,
    DEFAULT_ROLE_MODEL_ALLOWLIST,
    PROBE_MARKER,
    FreeLLMAPICancelled,
    FreeLLMAPIClient,
    FreeLLMAPIConfig,
    load_config,
)
from jcode_runtime import PermanentRuntimeError, RuntimeUnavailable, TransientRuntimeError



class QuotaLedgerRequestCountTests(unittest.TestCase):
    def test_note_call_counts_http_request_attempts(self):
        ledger = QuotaLedger()
        ledger.note_call("freellmapi", tokens=9, requests=4)
        row = ledger._row("freellmapi")
        self.assertEqual(row["calls_today"], 4)
        self.assertEqual(row["calls_this_minute"], 4)
        self.assertEqual(row["tokens_today"], 9)

    def test_legacy_note_call_defaults_to_one_request(self):
        ledger = QuotaLedger()
        ledger.note_call("freellmapi", tokens=2)
        row = ledger._row("freellmapi")
        self.assertEqual(row["calls_today"], 1)
        self.assertEqual(row["calls_this_minute"], 1)

class _Response:
    def __init__(self, payload, headers=None, status=200):
        self._payload = json.dumps(payload).encode("utf-8")
        self.headers = headers or {}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


def _models(*rows):
    if rows:
        return {"object": "list", "data": list(rows)}
    return {
        "object": "list",
        "data": [
            {"id": "qwen3:8b", "execution_status": "ready", "platforms": ["custom"]},
            {"id": "qwen2.5-coder:3b", "execution_status": "ready", "platforms": ["custom"]},
            {"id": "openai/gpt-4.1", "execution_status": "ready", "platforms": ["openai"]},
        ],
    }


def _completion(content="public response"):
    return {
        "id": "chatcmpl-public-test",
        "model": "qwen3:8b",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


def _config(**overrides):
    values = {
        "enabled": True,
        "base_url": DEFAULT_BASE_URL,
        "api_key_env": "VEO_FREELLMAPI_API_KEY",
        "role_model_allowlist": DEFAULT_ROLE_MODEL_ALLOWLIST,
        "allowed_classifications": ("PUBLIC",),
        "timeout_seconds": 45,
        "max_retries": 1,
        "retry_delay_seconds": 0,
        "max_tokens": 512,
    }
    values.update(overrides)
    return FreeLLMAPIConfig(**values)


def _load_payload(payload):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gateway.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_config(path)


class FreeLLMAPIConfigTests(unittest.TestCase):
    def test_workspace_config_is_enabled_and_public_only(self):
        config = load_config(Path(__file__).with_name("freellmapi.config.json"))
        self.assertTrue(config.enabled)
        self.assertEqual(config.base_url, DEFAULT_BASE_URL)
        self.assertEqual(config.allowed_classifications, ("PUBLIC",))
        self.assertEqual(set(config.role_model_allowlist), {"planning", "research", "qa"})

    def test_config_rejects_literal_api_keys_and_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "unsupported FreeLLMAPI configuration fields"):
            _load_payload({"enabled": False, "api_key": "must-not-be-accepted"})

    def test_config_rejects_non_public_classification(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            _load_payload({"enabled": False, "allowed_classifications": ["PUBLIC", "INTERNAL"]})

    def test_base_url_must_be_loopback_v1_with_port(self):
        urls = (
            "https://127.0.0.1:3001/v1",
            "http://gateway.example:3001/v1",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:3001/v1?x=1",
            "http://127.0.0.1/v1",
        )
        for url in urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                _load_payload({"base_url": url})

    def test_enabled_config_loaded_from_disk_can_build_a_client(self):
        """Regression: load_config yields a MappingProxyType, not a dict."""
        config = _load_payload({"enabled": True, "base_url": DEFAULT_BASE_URL})
        client = FreeLLMAPIClient(config)
        self.assertEqual(client.role_model_allowlist["coding"], ("qwen2.5-coder:3b",))
        self.assertEqual(client.spec.name, "freellmapi")

    def test_config_rejects_models_outside_the_approved_model_allowlist(self):
        with self.assertRaisesRegex(ValueError, "approved model IDs"):
            _load_payload({"role_model_allowlist": {"planning": ["openai/gpt-4.1"]}})


class FreeLLMAPIClientTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.client = FreeLLMAPIClient(_config(), telemetry=self.events.append)
        self.env = mock.patch.dict("os.environ", {"VEO_FREELLMAPI_API_KEY": "test-unified-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_disabled_gateway_cannot_be_used(self):
        with self.assertRaisesRegex(RuntimeUnavailable, "disabled"):
            FreeLLMAPIClient(_config(enabled=False))

    def test_default_private_classification_blocks_before_network_and_logs_no_prompt(self):
        private_prompt = "company task: do not disclose this"
        for classification in ("PRIVATE", "INTERNAL", "CONFIDENTIAL", "RESTRICTED", None):
            with self.subTest(classification=classification):
                with mock.patch("freellmapi_adapter._open_direct") as opener:
                    with self.assertRaisesRegex(PermanentRuntimeError, "blocks non-public"):
                        kwargs = {}
                        if classification is not None:
                            kwargs["data_classification"] = classification
                        self.client.chat(private_prompt, "planning", "qwen3:8b", **kwargs)
                opener.assert_not_called()
                self.assertEqual(self.events[-1]["outcome"], "blocked")
        self.assertNotIn(private_prompt, json.dumps(self.events))
        self.assertNotIn("test-unified-key", json.dumps(self.events))

    def test_company_task_cannot_be_relabelled_public_by_source_field(self):
        with mock.patch("freellmapi_adapter._open_direct") as opener:
            with self.assertRaisesRegex(PermanentRuntimeError, "blocks non-public"):
                self.client.chat(
                    "private task text",
                    "planning",
                    "qwen3:8b",
                    data_classification="PUBLIC",
                    source="task",
                )
        opener.assert_not_called()

    def test_role_and_model_allowlist_reject_before_network(self):
        with mock.patch("freellmapi_adapter._open_direct") as opener:
            with self.assertRaisesRegex(PermanentRuntimeError, "role"):
                self.client.chat(
                    "public prompt",
                    "unlisted",
                    "qwen3:8b",
                    data_classification="PUBLIC",
                    source="public_reference",
                )
            with self.assertRaisesRegex(PermanentRuntimeError, "model"):
                self.client.chat(
                    "public prompt",
                    "planning",
                    "qwen2.5-coder:3b",
                    data_classification="PUBLIC",
                    source="public_reference",
                )
        opener.assert_not_called()

    def test_models_endpoint_requires_environment_key_without_leaking_it(self):
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch(
            "freellmapi_adapter._open_direct"
        ) as opener:
            with self.assertRaisesRegex(RuntimeUnavailable, "VEO_FREELLMAPI_API_KEY"):
                self.client.list_models()
        opener.assert_not_called()
        self.assertNotIn("test-unified-key", str(self.client.descriptor()))

    def test_health_and_models_use_only_expected_loopback_paths(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[
                _Response({"status": "ok"}),
                _Response(_models()),
            ],
        ) as opener:
            self.assertEqual(self.client.health(), {"status": "ok"})
            models = self.client.list_models()
        self.assertEqual(models, ["qwen2.5-coder:3b", "qwen3:8b"])
        requests = [call.args[0] for call in opener.call_args_list]
        self.assertEqual(requests[0].full_url, "http://127.0.0.1:3001/api/ping")
        self.assertEqual(requests[1].full_url, "http://127.0.0.1:3001/v1/models")
        self.assertIsNone(requests[0].get_header("Authorization"))
        self.assertEqual(requests[1].get_header("Authorization"), "Bearer test-unified-key")

    def test_health_rejects_non_ok_json_status(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            return_value=_Response({"status": "unavailable"}),
        ):
            with self.assertRaisesRegex(RuntimeUnavailable, "/api/ping response is not ok"):
                self.client.health()

    def test_models_endpoint_filters_to_allowlisted_ready_models(self):
        response = _models(
            {"id": "qwen3:8b", "execution_status": "ready"},
            {"id": "qwen2.5-coder:3b", "execution_status": "needsKey"},
            {"id": "qwen3.8:27b", "execution_status": "ready"},
            {"id": "qwen2.5-coder:3b", "available": 0},
        )
        with mock.patch("freellmapi_adapter._open_direct", return_value=_Response(response)):
            self.assertEqual(self.client.list_models(), ["qwen3:8b"])

    def test_public_chat_sends_allowlisted_model_and_emits_redacted_telemetry(self):
        prompt = "A public question from a public reference."
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[
                _Response(_models()),
                _Response(_completion(), status=200),
            ],
        ) as opener:
            result = self.client.chat(
                prompt,
                "planning",
                "qwen3:8b",
                data_classification="PUBLIC",
                source="public_reference",
            )
        request = opener.call_args_list[1].args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://127.0.0.1:3001/v1/chat/completions")
        self.assertEqual(body["model"], "qwen3:8b")
        self.assertEqual(body["messages"][0]["content"], prompt)
        self.assertEqual(result["text"], "public response")
        self.assertEqual(result["model"], "qwen3:8b")
        self.assertEqual(result["session_id"], "chatcmpl-public-test")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(self.events[-1]["outcome"], "success")
        self.assertEqual(self.events[-1]["session_id"], "chatcmpl-public-test")
        serialized = json.dumps(self.events[-1])
        self.assertNotIn(prompt, serialized)
        self.assertNotIn("public response", serialized)
        self.assertNotIn("test-unified-key", serialized)

    def test_loopback_transport_bypasses_configured_environment_proxies(self):
        request = urllib.request.Request("http://127.0.0.1:3001/api/ping")
        opener = mock.Mock()
        opener.open.return_value = _Response({"status": "ok"})
        with mock.patch.dict("os.environ", {"HTTPS_PROXY": "http://proxy.example:8080"}):
            with mock.patch(
                "urllib.request.ProxyHandler", wraps=urllib.request.ProxyHandler
            ) as proxy_handler, mock.patch(
                "urllib.request.build_opener", return_value=opener
            ):
                from freellmapi_adapter import _open_direct

                with _open_direct(request, 3) as response:
                    self.assertEqual(json.loads(response.read()), {"status": "ok"})
        proxy_handler.assert_called_once_with({})
        opener.open.assert_called_once_with(request, timeout=3)

    def test_unready_model_is_not_sent_for_inference(self):
        response = _models({"id": "qwen3:8b", "execution_status": "needsKey"})
        with mock.patch(
            "freellmapi_adapter._open_direct",
            return_value=_Response(response),
        ) as opener:
            with self.assertRaisesRegex(RuntimeUnavailable, "not ready"):
                self.client.chat(
                    "public prompt",
                    "planning",
                    "qwen3:8b",
                    data_classification="PUBLIC",
                    source="public_reference",
                )
        self.assertEqual(opener.call_count, 1)

    def test_transient_failure_retries_with_bounded_attempts(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[socket.timeout(), _Response({"status": "ok"})],
        ) as opener:
            self.assertEqual(self.client.health(), {"status": "ok"})
        self.assertEqual(opener.call_count, 2)

    def test_failed_retry_telemetry_reports_actual_attempts(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[socket.timeout(), socket.timeout()],
        ):
            with self.assertRaises(TransientRuntimeError):
                self.client.chat(
                    "public prompt",
                    "planning",
                    "qwen3:8b",
                    data_classification="PUBLIC",
                    source="public_reference",
                )
        self.assertEqual(self.events[-1]["outcome"], "error")
        self.assertEqual(self.events[-1]["attempts"], 2)

    def test_permanent_http_error_does_not_retry_or_leak_error_text(self):
        secret_like_error = "test-unified-key"
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=urllib.error.HTTPError(
                "http://127.0.0.1:3001/health", 401, secret_like_error, None, None
            ),
        ) as opener:
            with self.assertRaises(PermanentRuntimeError) as caught:
                self.client.health()
        self.assertEqual(opener.call_count, 1)
        self.assertNotIn(secret_like_error, str(caught.exception))


class _RawResponse:
    """A non-JSON 200 response, like the gateway's SPA fallback route."""

    def __init__(self, body=b"<!doctype html><html></html>", status=200):
        self._body = body
        self.headers = {"Content-Type": "text/html"}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _horde_catalog(count=29, upstream="ai_horde"):
    rows = [
        {"id": "qwen3:8b", "execution_status": "ready", "platforms": [upstream]},
        {"id": "qwen2.5-coder:3b", "execution_status": "ready", "platforms": [upstream]},
        {"id": "openai/gpt-4.1", "execution_status": "ready", "platforms": [upstream]},
    ]
    for index in range(max(0, count - len(rows))):
        rows.append(
            {
                "id": "horde/model-%02d" % index,
                "execution_status": "ready",
                "platforms": [upstream],
            }
        )
    return {"object": "list", "data": rows}


class FreeLLMAPICatalogTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.client = FreeLLMAPIClient(_config(), telemetry=self.events.append)
        self.env = mock.patch.dict("os.environ", {"VEO_FREELLMAPI_API_KEY": "test-unified-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_catalog_groups_many_models_into_one_upstream_backend(self):
        with mock.patch(
            "freellmapi_adapter._open_direct", return_value=_Response(_horde_catalog(29))
        ):
            catalog = self.client.catalog()
        self.assertEqual(catalog["model_count"], 29)
        # 29 rows behind one upstream are one backend, not 29 providers.
        self.assertEqual(catalog["backend_count"], 1)
        self.assertEqual(list(catalog["upstreams"]), ["ai_horde"])
        self.assertEqual(catalog["allowed_ready_models"], ["qwen2.5-coder:3b", "qwen3:8b"])

    def test_catalog_counts_distinct_upstreams_separately(self):
        payload = {
            "data": [
                {"id": "qwen3:8b", "execution_status": "ready", "platforms": ["ai_horde"]},
                {"id": "qwen2.5-coder:3b", "execution_status": "ready", "platforms": ["ollama"]},
            ]
        }
        with mock.patch("freellmapi_adapter._open_direct", return_value=_Response(payload)):
            catalog = self.client.catalog()
        self.assertEqual(catalog["backend_count"], 2)

    def test_catalog_with_no_data_list_is_refused(self):
        with mock.patch("freellmapi_adapter._open_direct", return_value=_Response({"object": "list"})):
            with self.assertRaisesRegex(RuntimeUnavailable, "no data list"):
                self.client.catalog()

    def test_empty_catalog_yields_no_routes_and_no_invention(self):
        with mock.patch("freellmapi_adapter._open_direct", return_value=_Response({"data": []})):
            self.assertEqual(self.client.catalog()["model_count"], 0)
            self.assertEqual(self.client.list_models(), [])

    def test_spa_fallback_on_a_health_path_is_not_parsed_as_health(self):
        """`/health` returns the dashboard HTML with 200; that is not health."""
        with mock.patch("freellmapi_adapter._open_direct", return_value=_RawResponse()):
            with self.assertRaisesRegex(PermanentRuntimeError, "invalid JSON"):
                self.client.health()


class FreeLLMAPIExecuteContractTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.quota = free_providers.QuotaLedger()
        self.breaker = free_providers.CircuitBreaker(failure_threshold=2, cooldown_seconds=0.05)
        self.client = FreeLLMAPIClient(
            _config(), telemetry=self.events.append, quota=self.quota, breaker=self.breaker
        )
        self.env = mock.patch.dict("os.environ", {"VEO_FREELLMAPI_API_KEY": "test-unified-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_execute_refuses_every_non_public_classification_before_network(self):
        for classification in ("PRIVATE", "CONFIDENTIAL", "INTERNAL", "RESTRICTED", ""):
            with self.subTest(classification=classification):
                with mock.patch("freellmapi_adapter._open_direct") as opener:
                    with self.assertRaisesRegex(PermanentRuntimeError, "PUBLIC data only"):
                        self.client.execute(
                            "company secret plan",
                            tempfile.gettempdir(),
                            "qwen3:8b",
                            data_classification=classification,
                        )
                opener.assert_not_called()

    def test_execute_defaults_to_private_and_so_never_sends(self):
        with mock.patch("freellmapi_adapter._open_direct") as opener:
            with self.assertRaisesRegex(PermanentRuntimeError, "PUBLIC data only"):
                self.client.execute("unclassified task", tempfile.gettempdir(), "qwen3:8b")
        opener.assert_not_called()

    def test_execute_public_returns_the_worker_contract_shape(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[_Response(_models()), _Response(_completion(), status=200)],
        ) as opener:
            result = self.client.execute(
                "public question",
                tempfile.gettempdir(),
                "qwen2.5-coder:3b",
                data_classification="PUBLIC",
            )
        body = json.loads(opener.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertFalse(body["stream"], "streaming is not verified, so it is never requested")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["data_classification"], "PUBLIC")
        self.assertEqual(result["class"], "OPTIONAL_GATEWAY")
        self.assertFalse(result["paid"])
        self.assertFalse(result["streaming"])
        self.assertEqual(result["provider"], "freellmapi")
        self.assertTrue(result["request_id"])

    def test_each_request_carries_a_distinct_trace_id(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[
                _Response(_models()),
                _Response(_completion()),
                _Response(_models()),
                _Response(_completion()),
            ],
        ):
            first = self.client.execute(
                "public one", tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC"
            )
            second = self.client.execute(
                "public two", tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC"
            )
        self.assertNotEqual(first["request_id"], second["request_id"])
        self.assertEqual(self.events[-1]["request_id"], second["request_id"])
        self.assertNotIn("public two", json.dumps(self.events))

    def test_probe_without_a_key_is_blocked_without_any_transport(self):
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch(
            "freellmapi_adapter._open_direct"
        ) as opener:
            state = self.client.probe("qwen3:8b")
        opener.assert_not_called()
        self.assertFalse(state["ok"])
        self.assertFalse(state["available"])
        self.assertIn("VEO_FREELLMAPI_API_KEY", state["error"])

    def test_probe_reports_a_real_marker_answer(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[
                _Response(_models()),
                _Response(_models()),
                _Response(_completion(PROBE_MARKER), status=200),
            ],
        ):
            state = self.client.probe("qwen3:8b")
        self.assertTrue(state["ok"])
        self.assertTrue(state["marker_ok"])
        # The fixture rows declare platforms "custom" and "openai": two backends,
        # not three models.
        self.assertEqual(state["catalog"]["backend_count"], 2)

    def test_quota_exhaustion_blocks_before_transport(self):
        client = FreeLLMAPIClient(
            _config(requests_per_minute=1), quota=self.quota, breaker=self.breaker
        )
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[_Response(_models()), _Response(_completion())],
        ):
            client.execute("public", tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC")
        with mock.patch("freellmapi_adapter._open_direct") as opener:
            with self.assertRaisesRegex(RuntimeUnavailable, "quota exhausted"):
                client.execute(
                    "public again", tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC"
                )
        opener.assert_not_called()

    def test_rate_limit_is_counted_and_retried(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[
                _Response(_models()),
                urllib.error.HTTPError("http://127.0.0.1:3001/v1/chat/completions", 429, "slow down", None, None),
                _Response(_completion()),
            ],
        ):
            result = self.client.execute(
                "public", tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC"
            )
        self.assertEqual(result["text"], "public response")
        self.assertGreaterEqual(self.quota.snapshot("freellmapi")["rate_limited"], 1)

    def test_server_error_is_transient_and_bounded(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[
                _Response(_models()),
                urllib.error.HTTPError("http://x", 503, "down", None, None),
                urllib.error.HTTPError("http://x", 503, "down", None, None),
            ],
        ) as opener:
            with self.assertRaises(TransientRuntimeError):
                self.client.execute(
                    "public", tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC"
                )
        self.assertEqual(opener.call_count, 3, "one discovery + max_retries+1 chat attempts")

    def test_timeout_is_transient_and_does_not_leak_the_prompt(self):
        prompt = "public question with no secrets"
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[_Response(_models()), socket.timeout(), socket.timeout()],
        ):
            with self.assertRaises(TransientRuntimeError) as caught:
                self.client.execute(prompt, tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC")
        self.assertIn("timed out", str(caught.exception))
        self.assertNotIn(prompt, str(caught.exception))
        self.assertNotIn("test-unified-key", str(caught.exception))

    def test_401_is_permanent_counts_once_and_hides_the_key(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=urllib.error.HTTPError("http://x", 401, "test-unified-key", None, None),
        ) as opener:
            with self.assertRaises(PermanentRuntimeError) as caught:
                self.client.execute(
                    "public", tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC"
                )
        self.assertEqual(opener.call_count, 1)
        self.assertEqual(self.breaker.state_of("freellmapi"), "CLOSED")
        self.assertNotIn("test-unified-key", str(caught.exception))

    def test_403_is_permanent_and_not_retried(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=urllib.error.HTTPError("http://x", 403, "forbidden", None, None),
        ) as opener:
            with self.assertRaises(PermanentRuntimeError):
                self.client.execute(
                    "public", tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC"
                )
        self.assertEqual(opener.call_count, 1, "403 must never be retried")

    def test_breaker_opens_after_repeated_failures_and_recovers_half_open(self):
        failures = [
            urllib.error.HTTPError("http://x", 503, "down", None, None)
            for _ in range(2 * 2)
        ]
        with mock.patch("freellmapi_adapter._open_direct", side_effect=failures):
            for _ in range(2):
                with self.assertRaises(TransientRuntimeError):
                    self.client.execute(
                        "public", tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC"
                    )
        self.assertEqual(self.breaker.state_of("freellmapi"), "OPEN")
        with mock.patch("freellmapi_adapter._open_direct") as opener:
            with self.assertRaisesRegex(TransientRuntimeError, "circuit is open"):
                self.client.execute(
                    "public", tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC"
                )
        opener.assert_not_called()
        time.sleep(0.06)
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[_Response(_models()), _Response(_completion())],
        ):
            self.client.execute(
                "public", tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC"
            )
        self.assertEqual(self.breaker.state_of("freellmapi"), "CLOSED")

    def test_cancellation_stops_before_transport(self):
        cancel = threading.Event()
        cancel.set()
        with mock.patch("freellmapi_adapter._open_direct") as opener:
            with self.assertRaises(FreeLLMAPICancelled):
                self.client.execute(
                    "public",
                    tempfile.gettempdir(),
                    "qwen3:8b",
                    data_classification="PUBLIC",
                    cancel=cancel,
                )
        opener.assert_not_called()
        self.assertEqual(self.breaker.state_of("freellmapi"), "CLOSED")

    def test_cancellation_between_retries_is_honoured(self):
        cancel = threading.Event()

        def transport(_request, timeout):
            cancel.set()
            raise socket.timeout()

        with mock.patch("freellmapi_adapter._open_direct", side_effect=transport) as opener:
            with self.assertRaises((FreeLLMAPICancelled, TransientRuntimeError)):
                self.client.execute(
                    "public",
                    tempfile.gettempdir(),
                    "qwen3:8b",
                    data_classification="PUBLIC",
                    cancel=cancel,
                )
        self.assertEqual(opener.call_count, 1)


class FreeLLMAPIRouteStatusTests(unittest.TestCase):
    """Routing must reflect quota, circuit and measured latency, not the catalog."""

    def setUp(self):
        self.quota = free_providers.QuotaLedger()
        self.breaker = free_providers.CircuitBreaker(failure_threshold=1, cooldown_seconds=60)
        self.client = FreeLLMAPIClient(
            _config(requests_per_minute=1), quota=self.quota, breaker=self.breaker
        )
        self.env = mock.patch.dict("os.environ", {"VEO_FREELLMAPI_API_KEY": "test-unified-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_healthy_gateway_is_routable_and_reports_no_latency_yet(self):
        status = self.client.route_status()
        self.assertTrue(status["routable"])
        self.assertIsNone(status["reason"])
        self.assertIsNone(status["latency_seconds"], "never invent a latency")
        self.assertEqual(status["circuit"], "CLOSED")

    def test_open_circuit_removes_the_gateway_from_routing(self):
        self.breaker.record_failure("freellmapi", "503")
        status = self.client.route_status()
        self.assertFalse(status["routable"])
        self.assertEqual(status["reason"], "CIRCUIT_OPEN")

    def test_exhausted_quota_removes_the_gateway_from_routing(self):
        self.quota.note_call("freellmapi", 5)
        status = self.client.route_status()
        self.assertFalse(status["routable"])
        self.assertIn("QUOTA", status["reason"])

    def test_latency_comes_only_from_a_real_call(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[_Response(_models()), _Response(_completion())],
        ):
            self.client.execute(
                "public", tempfile.gettempdir(), "qwen3:8b", data_classification="PUBLIC"
            )
        self.assertIsNotNone(self.client.latency_seconds)
        self.assertGreaterEqual(self.client.latency_seconds, 0.0)
        self.assertEqual(self.client.route_status()["latency_seconds"], self.client.latency_seconds)

    def test_probe_result_is_recorded_as_inference_evidence(self):
        self.assertIsNone(self.client.last_probe)
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[
                _Response(_models()),
                _Response(_models()),
                _Response(_completion(PROBE_MARKER)),
            ],
        ):
            self.client.probe("qwen3:8b")
        self.assertTrue(self.client.last_probe["ok"])
        self.assertTrue(self.client.last_probe["marker_ok"])


if __name__ == "__main__":
    unittest.main()
