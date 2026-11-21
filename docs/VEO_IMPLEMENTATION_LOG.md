# VEO implementation log
Updated: 2026-11-21

## Direction
Reuse a mature agent runtime; do not build a bespoke company-agent framework. VEO differentiates through verified free-model routing, quota awareness, fallback, capability tests, and Windows usability.

## Windows runtime validation
- Official jcode v1.0.1 Windows x86_64 tarball downloaded; SHA256 matched official SHA256SUMS.
- Binary installed under %LOCALAPPDATA%\VEO\bin; version verified as v1.0.1.
- Configured no-key profile ollama-local at http://127.0.0.1:11434/v1 for qwen3.5:9b-64k with 65,536 context.
- Direct jcode inference returned JCODE_OLLAMA_OK.
- Read tool returned README heading # jcode; shell tool returned VEO_JCODE_SHELL_OK.
- auth-test --no-tool-smoke failed by probing default openrouter.env (401); direct named-profile inference succeeded. Investigate auth-test separately.
- Source caveat: local master is 1 ahead and 12,356 behind origin/master; Cargo.toml references vendor/ratatui-jcode, absent locally but present upstream. Do not change Cargo.toml to mask this.
- Use verified prebuilt Windows binary until source history and sparse checkout are reconciled.

## Next gates
- Diagnose named-profile auth-test behavior.
- Reconcile source checkout without overwriting pending work.
- Validate model routing, fallback, quotas, tool calls, and real Windows tasks.

## Additional validation (2026-11-21)
- Agent write/read smoke: created a temporary file with exact marker VEO_WRITE_READ_OK, read it back, then removed it; verified the file no longer exists.
- Provider registry exposes Claude, OpenAI, OpenRouter, Gemini, Ollama; registry presence is not proof of working credentials.
- Gemini Code Assist individual OAuth flow now reports unsupported; OpenRouter probe returns HTTP 401 missing bearer token.
- Re-ran AI_COMPANY/test_model_fleet.py: 6 passed in 0.02s.
- Ollama inventory confirms qwen3.5:9b-64k, qwen3.5:9b, qwen3:8b, qwen2.5-coder:3b-64k, qwen2.5-coder:3b, and qwen2.5-coder:1.5b-base installed locally.
- qwen2.5-coder:3b-64k exact-output inference smoke passed (CODER_MODEL_OK).
- qwen2.5-coder:3b-64k tool smoke emitted a bash tool-call JSON request but did  not execute it in headless run; do not mark coding-model tool use as passed. qwen3.5:9b-64k write/read tool cycle was executed and verified by jcode.
- Hermes Agent v0.23.3 is already installed natively; hermes doctor confirms Python 3.14.9, SQLite, config, built-in memory, Chromium, Git, ripgrep, Docker, and Node are available.
- Hermes config already defines a local OpenAI-compatible Ollama endpoint at 127.0.0.1:11434/v1 and qwen3.5:9b-64k; provider is present but interactive Hermes session not yet verified.
- Hermes doctor reports one setup issue for missing API keys/full optional tool access; several integrations are unavailable without their own dependencies or credentials.
