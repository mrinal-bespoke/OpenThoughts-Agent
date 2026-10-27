# Stage 0 — strict contract and audit

Add only `eval/autoeval32k` and focused tests. Validate every manifest field;
reject floating revisions, duplicate tasks, unknown options and invalid budgets.
Hash protocol/model/tasks/runtime pins into run identity. Preserve raw files and
report missing, unscored, zero and nonzero outcomes separately. Tests must cover
malformed evidence, duplicate outcomes and partial coverage. No legacy defaults
are edited; normal imports perform no network, scheduler or registry operations.

Gate: focused CPU tests, relevant eval regression tests, legacy diff empty.

CPU gate passed: 15 protocol/audit tests; 37 neighboring wrapper tests pass.
Existing Harbor JobConfig accepted attempts=1, retries=0, concurrency=1 and both
explicit adapter import paths. Legacy launchers/configs are unchanged.
Repository mechanical lint entrypoint is absent; supplementary Ruff check passed.
