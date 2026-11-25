"""Regression tests for FreeLLMAPI tool calling (mocked HTTP only)."""
import json
import socket
import sys
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import free_providers
from freellmapi_adapter import (
    DEFAULT_BASE_URL,
    MAX_TOOL_ROUNDS,
    MAX_TOOL_CALLS_PER_ROUND,
    MAX_TOOL_CALLS_TOTAL,
    TOOL_PROFILE,
    FreeLLMAPIClient,
    FreeLLMAPIConfig,
    FreeLLMAPIToolDenied,
    FreeLLMAPIToolUnsupported,
    _validate_tool_arguments,
    _prevalidate_tool_batch,
)
from jcode_runtime import PermanentRuntimeError, RuntimeUnavailable, TransientRuntimeError

MODEL = "qwen3-8b"
ECHO_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "repeat": {"type": "integer"},
        "mode": {"type": "string", "enum": ["fast", "slow"]},
    },
    "required": ["text"],
    "additionalProperties": False,
}
ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "veo_echo",
        "description": "Echo text back. Harmless.",
        "parameters": ECHO_SCHEMA,
    },
}
DELETE_TOOL = {
    "type": "function",
    "function": {
        "name": "veo_delete",
        "description": "Dangerous tool that must never run without a grant.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}


class _Response:
    def __init__(self, payload, status=200):
        self._payload = json.dumps(payload).encode("utf-8")
        self.headers = {}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


def _models(tool_capable=True):
    parameters = ["temperature", "max_tokens", "stream"]
    if tool_capable:
        parameters.append("tools")
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL,
                "execution_status": "ready",
                "supported_parameters": parameters,
            }
        ],
    }


def _tool_call(call_id="call_abc", name="veo_echo", arguments='{"text": "VEO_TOOL_OK"}'):
    return [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
    ]


def _tool_round(calls, content=None, finish="tool_calls", response_id="chatcmpl-tools"):
    return {
        "id": response_id,
        "model": MODEL,
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "tool_calls": calls},
                "finish_reason": finish,
            }
        ],
        "usage": {"total_tokens": 11},
    }


def _text_round(text="final answer", response_id="chatcmpl-final"):
    return {
        "id": response_id,
        "model": MODEL,
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 7},
    }


def _config(**overrides):
    values = {
        "enabled": True,
        "base_url": DEFAULT_BASE_URL,
        "api_key_env": "VEO_FREELLMAPI_API_KEY",
        "role_model_allowlist": {"research": (MODEL,)},
        "allowed_classifications": ("PUBLIC",),
        "timeout_seconds": 45,
        "max_retries": 1,
        "retry_delay_seconds": 0,
        "max_tokens": 512,
    }
    values.update(overrides)
    return FreeLLMAPIConfig(**values)


class ToolCallingTestBase(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.client = FreeLLMAPIClient(_config(), telemetry=self.events.append)
        self.env = mock.patch.dict("os.environ", {"VEO_FREELLMAPI_API_KEY": "test-unified-key"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.calls = []
        self.handlers = {"veo_echo": self._echo, "veo_delete": self._delete}

    def _fresh_client(self):
        """A new breaker/quota per loop iteration: permanent errors trip it."""
        self.breaker_client = FreeLLMAPIClient(_config(), telemetry=self.events.append)
        return self.breaker_client

    def _echo(self, arguments):
        self.calls.append(("veo_echo", arguments))
        return {"echo": arguments.get("text")}

    def _delete(self, arguments):
        self.calls.append(("veo_delete", arguments))
        return "should never happen"

    def run_tool(self, responses, **kwargs):
        options = {
            "tools": [ECHO_TOOL],
            "handlers": self.handlers,
            "granted": {"veo_echo"},
            "data_classification": "PUBLIC",
            "source": "public_reference",
        }
        options.update(kwargs)
        with mock.patch("freellmapi_adapter._open_direct", side_effect=responses) as opener:
            result = self.client.run_tools("public prompt", "research", MODEL, **options)
        return result, opener


class ToolRequestTests(ToolCallingTestBase):
    def test_request_carries_openai_tool_schema_and_choice(self):
        _, opener = self.run_tool([_Response(_models()), _Response(_text_round())])
        body = json.loads(opener.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(body["tools"], [ECHO_TOOL])
        self.assertNotIn("tool_choice", body, "auto is the gateway default; no pinned choice")
        self.assertEqual(body["model"], MODEL)
        self.assertFalse(body["stream"])

    def test_required_choice_is_pinned_on_the_first_round_only(self):
        result, opener = self.run_tool(
            [
                _Response(_models()),
                _Response(_tool_round(_tool_call())),
                _Response(_text_round()),
            ],
            tool_choice="required",
        )
        first = json.loads(opener.call_args_list[1].args[0].data.decode("utf-8"))
        after = json.loads(opener.call_args_list[2].args[0].data.decode("utf-8"))
        self.assertEqual(first["tool_choice"], "required")
        self.assertNotIn("tool_choice", after, "the round after a tool result stays auto")
        self.assertEqual(result["tools_executed"], 1)

    def test_invalid_tool_choice_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tool_choice"):
            self.client.run_tools(
                "public prompt",
                "research",
                MODEL,
                tools=[ECHO_TOOL],
                handlers=self.handlers,
                granted={"veo_echo"},
                data_classification="PUBLIC",
                source="public_reference",
                tool_choice="always",
            )

    def test_text_only_chat_never_sends_tools(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[_Response(_models()), _Response(_text_round("plain text"))],
        ) as opener:
            result = self.client.chat(
                "public prompt",
                "research",
                MODEL,
                data_classification="PUBLIC",
                source="public_reference",
            )
        body = json.loads(opener.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertNotIn("tools", body)
        self.assertNotIn("tool_choice", body)
        self.assertEqual(result["text"], "plain text")


class ToolPermissionTests(ToolCallingTestBase):
    def test_ungranted_tool_is_refused_before_any_network_access(self):
        with mock.patch("freellmapi_adapter._open_direct") as opener:
            with self.assertRaisesRegex(FreeLLMAPIToolDenied, "permission denied"):
                self.client.run_tools(
                    "public prompt",
                    "research",
                    MODEL,
                    tools=[ECHO_TOOL, DELETE_TOOL],
                    handlers=self.handlers,
                    granted={"veo_echo"},
                    data_classification="PUBLIC",
                    source="public_reference",
                )
        opener.assert_not_called()
        self.assertEqual(self.calls, [])

    def test_empty_grant_executes_nothing(self):
        with mock.patch("freellmapi_adapter._open_direct") as opener:
            with self.assertRaisesRegex(FreeLLMAPIToolDenied, "no tool permission granted"):
                self.client.run_tools(
                    "public prompt",
                    "research",
                    MODEL,
                    tools=[ECHO_TOOL],
                    handlers=self.handlers,
                    granted=set(),
                    data_classification="PUBLIC",
                    source="public_reference",
                )
        opener.assert_not_called()

    def test_model_invented_tool_is_denied_and_not_executed(self):
        responses = [
            _Response(_models()),
            _Response(_tool_round(_tool_call(name="rm_rf"))),
        ]
        with mock.patch("freellmapi_adapter._open_direct", side_effect=responses):
            with self.assertRaisesRegex(FreeLLMAPIToolDenied, "permission denied"):
                self.client.run_tools(
                    "public prompt",
                    "research",
                    MODEL,
                    tools=[ECHO_TOOL],
                    handlers=self.handlers,
                    granted={"veo_echo"},
                    data_classification="PUBLIC",
                    source="public_reference",
                )
        self.assertEqual(self.calls, [])

    def test_non_public_data_is_refused_before_any_tool_call(self):
        with mock.patch("freellmapi_adapter._open_direct") as opener:
            with self.assertRaisesRegex(PermanentRuntimeError, "blocks non-public"):
                self.client.run_tools(
                    "private task",
                    "research",
                    MODEL,
                    tools=[ECHO_TOOL],
                    handlers=self.handlers,
                    granted={"veo_echo"},
                )
        opener.assert_not_called()
        self.assertEqual(self.calls, [])


class ToolArgumentTests(ToolCallingTestBase):
    def test_arguments_mismatching_schema_are_refused_without_execution(self):
        bad = [
            '{"repeat": 2}',  # missing required "text"
            '{"text": 5}',  # wrong type
            '{"text": "ok", "extra": true}',  # additionalProperties false
            '{"text": "ok", "mode": "turbo"}',  # enum
            '{"text": "ok", "repeat": "2"}',  # integer, not string
            '{"text": "ok", "repeat": true}',  # bool is not an integer
            "not json",  # malformed arguments
            '["text"]',  # not an object
        ]
        for arguments in bad:
            with self.subTest(arguments=arguments):
                responses = [
                    _Response(_models()),
                    _Response(_tool_round(_tool_call(arguments=arguments))),
                ]
                client = self._fresh_client()
                with mock.patch("freellmapi_adapter._open_direct", side_effect=responses):
                    with self.assertRaises(PermanentRuntimeError):
                        client.run_tools(
                            "public prompt",
                            "research",
                            MODEL,
                            tools=[ECHO_TOOL],
                            handlers=self.handlers,
                            granted={"veo_echo"},
                            data_classification="PUBLIC",
                            source="public_reference",
                        )
        self.assertEqual(self.calls, [], "no handler ran for any invalid payload")

    def test_valid_arguments_are_passed_through_unmodified(self):
        result, _ = self.run_tool(
            [
                _Response(_models()),
                _Response(_tool_round(_tool_call(arguments='{"text": "VEO_TOOL_OK", "repeat": 2}'))),
                _Response(_text_round()),
            ]
        )
        self.assertEqual(self.calls, [("veo_echo", {"text": "VEO_TOOL_OK", "repeat": 2})])
        self.assertEqual(result["tool_calls"][0]["arguments"], {"text": "VEO_TOOL_OK", "repeat": 2})


    def test_nested_object_and_array_constraints_are_recursive(self):
        schema = {
            "type": "object",
            "properties": {
                "meta": {"type": "object", "required": ["label"], "properties": {
                    "label": {"type": "string"}, "count": {"type": "integer"}
                }, "additionalProperties": False},
                "values": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["meta", "values"],
            "additionalProperties": False,
        }
        self.assertEqual(_validate_tool_arguments(schema, {"meta": {"label": "x"}, "values": [1, 2]}),
                         {"meta": {"label": "x"}, "values": [1, 2]})
        with self.assertRaises(FreeLLMAPIToolDenied):
            _validate_tool_arguments(schema, {"meta": {"label": "x", "count": "wrong"}, "values": [1]})
        with self.assertRaises(FreeLLMAPIToolDenied):
            _validate_tool_arguments(schema, {"meta": {}, "values": [1]})

    def test_enum_comparison_is_type_aware(self):
        schema = {"type": "object", "properties": {"flag": {"type": "boolean", "enum": [True]}},
                  "required": ["flag"], "additionalProperties": False}
        self.assertEqual(_validate_tool_arguments(schema, {"flag": True}), {"flag": True})
        with self.assertRaises(FreeLLMAPIToolDenied):
            _validate_tool_arguments(schema, {"flag": 1})

    def test_unsupported_schema_keyword_is_rejected(self):
        schema = {"type": "object", "properties": {"text": {"type": "string", "pattern": "^ok$"}},
                  "required": ["text"]}
        with self.assertRaises(FreeLLMAPIToolDenied):
            _validate_tool_arguments(schema, {"text": "bad"})



    def test_batch_prevalidation_is_atomic_before_handlers(self):
        calls = [
            {"id": "one", "function": {"name": "safe", "arguments": "{\"value\":1}"}},
            {"id": "two", "function": {"name": "safe", "arguments": "{\"value\":\"bad\"}"}},
        ]
        side_effects = []
        schema = {"type": "object", "properties": {"value": {"type": "integer"}},
                  "required": ["value"], "additionalProperties": False}
        handlers = {"safe": lambda args: side_effects.append(args)}
        with self.assertRaises(FreeLLMAPIToolDenied):
            _prevalidate_tool_batch(calls, {"safe": schema}, handlers, [])
        self.assertEqual(side_effects, [])

    def test_duplicate_tool_call_ids_rejected_before_execution(self):
        call = {"id": "same", "function": {"name": "safe", "arguments": "{}"}}
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        with self.assertRaises(FreeLLMAPIToolDenied):
            _prevalidate_tool_batch([call, dict(call)], {"safe": schema}, {"safe": lambda _: None}, [])

    def test_call_count_limits_are_checked_before_execution(self):
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        calls = [{"id": str(i), "function": {"name": "safe", "arguments": "{}"}}
                 for i in range(MAX_TOOL_CALLS_PER_ROUND + 1)]
        with self.assertRaises(FreeLLMAPIToolDenied):
            _prevalidate_tool_batch(calls, {"safe": schema}, {"safe": lambda _: None}, [])
        with self.assertRaises(FreeLLMAPIToolDenied):
            _prevalidate_tool_batch([{"id": "next", "function": {"name": "safe", "arguments": "{}"}}],
                                    {"safe": schema}, {"safe": lambda _: None},
                                    [{"id": str(i)} for i in range(MAX_TOOL_CALLS_TOTAL)])

class ToolRoundTripTests(ToolCallingTestBase):
    def test_tool_call_id_and_result_round_trip(self):
        result, opener = self.run_tool(
            [
                _Response(_models()),
                _Response(_tool_round(_tool_call(call_id="call_keep_me"))),
                _Response(_text_round("echoed", response_id="chatcmpl-final")),
            ]
        )
        second = json.loads(opener.call_args_list[2].args[0].data.decode("utf-8"))
        roles = [message["role"] for message in second["messages"]]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        assistant = second["messages"][1]
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_keep_me")
        tool_message = second["messages"][2]
        self.assertEqual(tool_message["tool_call_id"], "call_keep_me")
        self.assertEqual(tool_message["name"], "veo_echo")
        self.assertIn("VEO_TOOL_OK", tool_message["content"])
        self.assertEqual(result["tool_calls"][0]["id"], "call_keep_me")
        self.assertEqual(result["tools_executed"], 1)
        self.assertEqual(result["tool_rounds"], 1)
        self.assertEqual(result["text"], "echoed")
        self.assertEqual(result["tool_profile"], TOOL_PROFILE)
        self.assertEqual(self.events[-1]["operation"], "chat.completions.tools")
        self.assertEqual(self.events[-1]["tool_calls"], 1)
        self.assertNotIn("public prompt", json.dumps(self.events))
        self.assertNotIn("test-unified-key", json.dumps(self.events))

    def test_two_sequential_tool_rounds_are_bounded_and_complete(self):
        result, _ = self.run_tool(
            [
                _Response(_models()),
                _Response(_tool_round(_tool_call(call_id="call_1"))),
                _Response(_tool_round(_tool_call(call_id="call_2"))),
                _Response(_text_round("finished")),
            ]
        )
        self.assertEqual([call["id"] for call in result["tool_calls"]], ["call_1", "call_2"])
        self.assertEqual(result["tool_rounds"], 2)
        self.assertEqual(result["text"], "finished")
        self.assertFalse(result["tool_budget_exhausted"])

    def test_round_budget_stops_execution_and_forces_a_text_answer(self):
        responses = [_Response(_models())]
        for index in range(MAX_TOOL_ROUNDS):
            responses.append(_Response(_tool_round(_tool_call(call_id="call_%d" % index))))
        responses.append(_Response(_tool_round(_tool_call(call_id="call_unexecuted"))))
        responses.append(_Response(_text_round("summary after budget")))
        result, opener = self.run_tool(responses, max_rounds=MAX_TOOL_ROUNDS)
        self.assertEqual(result["tools_executed"], MAX_TOOL_ROUNDS)
        self.assertNotIn("call_unexecuted", [call["id"] for call in result["tool_calls"]])
        final_body = json.loads(opener.call_args_list[-1].args[0].data.decode("utf-8"))
        self.assertEqual(final_body["tool_choice"], "none")
        self.assertEqual(result["text"], "summary after budget")
        self.assertFalse(result["tool_budget_exhausted"])

    def test_failing_tool_becomes_an_error_result_not_a_crash(self):
        def boom(_arguments):
            raise RuntimeError("tool exploded")

        result, _ = self.run_tool(
            [
                _Response(_models()),
                _Response(_tool_round(_tool_call())),
                _Response(_text_round("handled")),
            ],
            handlers={"veo_echo": boom, "veo_delete": self._delete},
        )
        self.assertFalse(result["tool_calls"][0]["ok"])
        self.assertIn("tool exploded", result["tool_calls"][0]["error"])
        self.assertEqual(result["text"], "handled")


class ToolFailureTests(ToolCallingTestBase):
    def test_malformed_tool_calls_do_not_crash(self):
        payloads = [
            {"id": "chatcmpl", "choices": [{"message": {"tool_calls": "nope"}}]},
            {"id": "chatcmpl", "choices": [{"message": {"tool_calls": [{}]}}]},
            {
                "id": "chatcmpl",
                "choices": [{"message": {"tool_calls": [{"id": "x", "function": {"name": "veo_echo"}}]}}],
            },
            {
                "id": "chatcmpl",
                "choices": [
                    {"message": {"tool_calls": [{"id": "x", "function": {"name": "veo_echo", "arguments": "{bad"}}]}}
                ],
            },
            {"choices": [{"message": "not an object"}]},
            {"choices": []},
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                responses = [_Response(_models()), _Response(payload)]
                client = self._fresh_client()
                with mock.patch("freellmapi_adapter._open_direct", side_effect=responses):
                    with self.assertRaises(PermanentRuntimeError):
                        client.run_tools(
                            "public prompt",
                            "research",
                            MODEL,
                            tools=[ECHO_TOOL],
                            handlers=self.handlers,
                            granted={"veo_echo"},
                            data_classification="PUBLIC",
                            source="public_reference",
                        )
        self.assertEqual(self.calls, [])

    def test_http_error_is_transient_and_bounded(self):
        responses = [
            _Response(_models()),
            urllib.error.HTTPError("http://x", 503, "down", None, None),
            urllib.error.HTTPError("http://x", 503, "down", None, None),
        ]
        with mock.patch("freellmapi_adapter._open_direct", side_effect=responses) as opener:
            with self.assertRaises(TransientRuntimeError):
                self.client.run_tools(
                    "public prompt",
                    "research",
                    MODEL,
                    tools=[ECHO_TOOL],
                    handlers=self.handlers,
                    granted={"veo_echo"},
                    data_classification="PUBLIC",
                    source="public_reference",
                )
        self.assertEqual(opener.call_count, 3)
        self.assertEqual(self.calls, [])

    def test_transport_timeout_is_transient(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[_Response(_models()), socket.timeout(), socket.timeout()],
        ):
            with self.assertRaises(TransientRuntimeError):
                self.client.run_tools(
                    "public prompt",
                    "research",
                    MODEL,
                    tools=[ECHO_TOOL],
                    handlers=self.handlers,
                    granted={"veo_echo"},
                    data_classification="PUBLIC",
                    source="public_reference",
                )

    def test_overall_deadline_stops_a_slow_tool_loop(self):
        client = FreeLLMAPIClient(_config(timeout_seconds=1))

        def slow(_arguments):
            threading.Event().wait(1.3)
            return "late"

        responses = [
            _Response(_models()),
            _Response(_tool_round(_tool_call())),
            _Response(_text_round()),
        ]
        with mock.patch("freellmapi_adapter._open_direct", side_effect=responses):
            with self.assertRaisesRegex(TransientRuntimeError, "timed out"):
                client.run_tools(
                    "public prompt",
                    "research",
                    MODEL,
                    tools=[ECHO_TOOL],
                    handlers={"veo_echo": slow},
                    granted={"veo_echo"},
                    data_classification="PUBLIC",
                    source="public_reference",
                )


class ToolCapabilityTests(ToolCallingTestBase):
    def test_model_without_advertised_tools_is_refused(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[_Response(_models(tool_capable=False))],
        ) as opener:
            with self.assertRaisesRegex(FreeLLMAPIToolUnsupported, "does not advertise"):
                self.client.run_tools(
                    "public prompt",
                    "research",
                    MODEL,
                    tools=[ECHO_TOOL],
                    handlers=self.handlers,
                    granted={"veo_echo"},
                    data_classification="PUBLIC",
                    source="public_reference",
                )
        self.assertEqual(opener.call_count, 1, "no completion is sent for an unverified model")
        self.assertEqual(self.calls, [])

    def test_missing_model_row_is_unknown_not_supported(self):
        payload = {"data": [{"id": "something-else", "execution_status": "ready", "supported_parameters": ["tools"]}]}
        with mock.patch("freellmapi_adapter._open_direct", return_value=_Response(payload)):
            with self.assertRaises(FreeLLMAPIToolUnsupported):
                self.client.run_tools(
                    "public prompt",
                    "research",
                    MODEL,
                    tools=[ECHO_TOOL],
                    handlers=self.handlers,
                    granted={"veo_echo"},
                    data_classification="PUBLIC",
                    source="public_reference",
                )

    def test_explicit_text_fallback_is_used_when_tools_are_unsupported(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[
                _Response(_models(tool_capable=False)),
                _Response(_models()),
                _Response(_text_round("text only")),
            ],
        ) as opener:
            result = self.client.run_tools(
                "public prompt",
                "research",
                MODEL,
                tools=[ECHO_TOOL],
                handlers=self.handlers,
                granted={"veo_echo"},
                data_classification="PUBLIC",
                source="public_reference",
                fallback_to_text=True,
            )
        self.assertEqual(result["text"], "text only")
        self.assertEqual(result["tool_calling"], "unsupported_by_model")
        self.assertEqual(result["tool_profile"], "none")
        chat_body = json.loads(opener.call_args_list[2].args[0].data.decode("utf-8"))
        self.assertNotIn("tools", chat_body)


class ExecuteToolPassthroughTests(ToolCallingTestBase):
    def test_execute_with_tools_reports_the_bounded_tool_profile(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[
                _Response(_models()),
                _Response(_tool_round(_tool_call())),
                _Response(_text_round("done")),
            ],
        ):
            result = self.client.execute(
                "public prompt",
                tempfile.gettempdir(),
                MODEL,
                data_classification="PUBLIC",
                tools=[ECHO_TOOL],
                handlers=self.handlers,
                granted={"veo_echo"},
            )
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["tool_profile"], TOOL_PROFILE)
        self.assertEqual(result["tools_executed"], 1)
        self.assertEqual(result["provider"], "freellmapi")

    def test_execute_without_tools_still_reports_no_tools(self):
        with mock.patch(
            "freellmapi_adapter._open_direct",
            side_effect=[_Response(_models()), _Response(_text_round("plain"))],
        ):
            result = self.client.execute(
                "public prompt",
                tempfile.gettempdir(),
                MODEL,
                data_classification="PUBLIC",
            )
        self.assertEqual(result["tool_profile"], "none")
        self.assertEqual(result["text"], "plain")

    def test_descriptor_reports_a_gated_tool_profile(self):
        descriptor = self.client.descriptor()
        self.assertEqual(descriptor["tool_profile"], TOOL_PROFILE)
        self.assertTrue(descriptor["tool_calling"]["permission_required"])
        self.assertTrue(descriptor["tool_calling"]["schema_validated"])
        self.assertNotIn("test-unified-key", json.dumps(descriptor))


class ToolQuotaTests(unittest.TestCase):
    def test_quota_and_breaker_gate_tool_calls_before_network(self):
        quota = free_providers.QuotaLedger()
        breaker = free_providers.CircuitBreaker(failure_threshold=1, cooldown_seconds=60)
        client = FreeLLMAPIClient(_config(requests_per_minute=1), quota=quota, breaker=breaker)
        with mock.patch.dict("os.environ", {"VEO_FREELLMAPI_API_KEY": "test-unified-key"}):
            quota.note_call("freellmapi", 1)
            with mock.patch("freellmapi_adapter._open_direct") as opener:
                with self.assertRaisesRegex(RuntimeUnavailable, "quota exhausted"):
                    client.run_tools(
                        "public prompt",
                        "research",
                        MODEL,
                        tools=[ECHO_TOOL],
                        handlers={"veo_echo": lambda a: a},
                        granted={"veo_echo"},
                        data_classification="PUBLIC",
                        source="public_reference",
                    )
            opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
