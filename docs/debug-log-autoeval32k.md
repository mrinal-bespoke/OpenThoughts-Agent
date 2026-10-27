# Debugging the TACC automatic evaluation path

## Runtime compatibility

The deployed harness has Harbor 0.8.1 but different source fingerprints from the
locally reviewed adapter, and it has no transformers distribution. The serving
environment already has transformers 5.12.1 and the proven vLLM build.

Use a private exact Harbor checkout with the existing harness dependencies;
perform tokenizer preparation in serving Python. No package installation or
shared runtime modification is needed if the import/schema gate passes.

## Cache reuse

A revision-named cache directory is not content proof. Verify each selected
model/tokenizer file against published immutable-revision metadata, using LFS
SHA256 or git blob SHA1 as appropriate. Stage private references only after all
hashes pass; reject corrupt bytes before starting serving. Preparation has an
explicit subprocess deadline. Four regression tests pass; combined gate 159.

## Intake deadline

The first metadata-only intake exited 124 at its internal 32-second bound; no
job was submitted. A staged read-only probe showed HF import at 0.41 seconds,
published revision metadata at 0.58 seconds, and transformers import at 15.04
seconds. Remove that unnecessary GPU-stack import from intake and download only
config/template metadata. Serving Python still verifies the actual loaded
tokenizer against the frozen hash before execution. Retain the partial intake
cache and use a fresh metadata-attempt directory; never overwrite evidence.

The existing TACC batch recipe also loads gcc/13.2.0 and cuda/12.8 and uses
enforce-eager. Carry these into the isolated allocation instead of introducing
an unqualified JIT/compiler configuration.
