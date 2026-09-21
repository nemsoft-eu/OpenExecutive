<!--
Thanks for contributing to Open Executive! Keep it to these three sections.
See .github/CONTRIBUTING.md for the full contribution guide.

Detail belongs in the commit message, not here. Open questions belong in the
review thread.
-->

## Problem

<!-- What is broken or missing, and how does it show up? -->

## Approach

<!-- What the change does. Call out anything non-obvious; keep it short. -->

## Checklist

- [ ] Working implementation — no stubs or TODO placeholders
- [ ] Tests added/updated for new behavior (`pytest packages/core/tests/unit/`)
- [ ] `ruff check` and `mypy` pass (`make lint`)
- [ ] UI builds if touched (`cd packages/ui && npm run build`)
- [ ] Eval scenarios added for a new agent or prompt change (if applicable)
- [ ] Architecture docs updated if a documented topic changed
      (see the "Architecture Docs" section in `CLAUDE.md`)
- [ ] No secrets, credentials, or personal data committed
