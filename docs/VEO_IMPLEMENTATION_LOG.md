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
