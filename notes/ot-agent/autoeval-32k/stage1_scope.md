# Stage 1 — TACC execution

Use `eval/autoeval32k` only; do not call the legacy listener or batch entrypoint.
Implement explicit settings, read-only planning, immutable bundle, exclusive
submission journal, all-user queue capacity, fresh run directory and strict
Harbor adapter. Stage task copies and verify their byte hashes. Bind the exact
checkpoint, tokenizer template and runtime versions. Preserve raw results and
check verifier evidence. Every created environment needs cleanup evidence.

CPU gates: interrupted submission cannot duplicate; missing scheduler visibility
does not free capacity; malformed or drifting files reject; old run untouched;
missing snapshot cannot build; serialized token overflow rejects before request;
effective Harbor schema matches. Live gate: separately owned synthetic smoke
with GPU BF16, generated response, compute Daytona command, agent and cleanup.

Not complete until live gate passes. Current access blocker prevents deployment
and live qualification. Concurrency remains 1. Jupiter is rejected by the CLI.
