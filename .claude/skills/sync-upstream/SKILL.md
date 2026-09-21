---
name: sync-upstream
description: Merge upstream SenteLabsAI/OpenExecutive into the nemsoft fork. Use when asked to sync with upstream, pull in upstream changes, check how far behind upstream we are, or resolve a fork/upstream merge. Encodes the remote-naming trap, the conflict hotspot map, the fork-invariant checklist, and the gate.
---

# sync-upstream

Merges upstream into `nemsoft/main` and opens a PR. Weekly cadence; upstream
lands roughly 35 commits a week, so the gap compounds fast if skipped.

## The remote names are inverted — read this first

```
origin   → github.com/SenteLabsAI/OpenExecutive   ← UPSTREAM (not ours)
nemsoft  → github.com/nemsoft-eu/OpenExecutive    ← OUR FORK
```

`origin` is **not** our repo. Every habit that says "push to origin" is wrong
here and would push fork commits at upstream. Always name the remote explicitly:
`git push nemsoft <branch>`, never a bare `git push` on a branch whose upstream
you have not checked. Never push to `origin` at all.

## Step 1 — report the delta (safe, read-only)

```bash
git -C /var/home/maarten/repos/openexecutive fetch --all --prune
git -C /var/home/maarten/repos/openexecutive rev-list --left-right --count origin/main...nemsoft/main
# output: "<upstream-only>\t<ours-only>"  — left is how far behind we are
```

Zero on the left means nothing to do. Stop and say so; do not create a worktree.

## Step 2 — the silent-divergence scan (do this BEFORE resolving anything)

Conflicts are the *less* dangerous half. The dangerous half is files upstream
changed that our fork never touched: those merge clean, with no marker and no
prompt, and can change behaviour silently.

```bash
BASE=$(git merge-base origin/main nemsoft/main)
# upstream-changed AND fork-untouched = merges clean = silent
comm -23 <(git diff --name-only $BASE origin/main -- 'packages/core/openexecutive/*.py' 'packages/ui/src' | sort) \
         <(git diff --name-only $BASE nemsoft/main -- 'packages/core/openexecutive/*.py' 'packages/ui/src' | sort)
```

**Rank that set by risk, not by diff size.** Read anything on an AUTH,
IDENTITY, GATE, or DELETE path first. Precedent: upstream's
`packages/ui/src/lib/allowlist.ts` was 6th by size and 1st by risk — it changed
sign-in from "the People roster is authoritative" to "roster UNION the
`ALLOWED_EMAILS` env var", because a fixture load does `DELETE FROM people`
(`cli/fixture_loader.py`) and was silently evicting the operator from their own
instance. Three files carried that change (`api/routes/auth.py`,
`ui/src/auth.ts`, `ui/src/lib/allowlist.ts`) and none of them conflicted.

Name every silent behaviour change you find in the PR body as its own item.

## Step 3 — worktree and baseline

Sibling worktree, never `.claude/worktrees/`:

```bash
git -C /var/home/maarten/repos/openexecutive worktree add -b chore/sync-upstream-<YYYY-MM-DD> \
    /var/home/maarten/repos/openexecutive-sync-upstream nemsoft/main
```

Copy the contents (not the directories) of `.claude/` and `.vscode/`, plus
`.env`, `.mcp.json`, `CLAUDE.local.md` if present. Then capture the baseline
**before merging**, so post-merge failures are attributable:

```bash
cd /var/home/maarten/repos/openexecutive-sync-upstream/packages/core && uv sync
env -u BACKEND_SHARED_SECRET -u OE_PUBLIC_DEPLOYMENT uv run pytest tests/unit/ -q
cd /var/home/maarten/repos/openexecutive-sync-upstream && make lint
```

Record the pass count. As of 2026-09-21 the baseline was green
(3698 passed, 1 skipped), so the post-merge bar is green, not "no new failures".

## Step 4 — merge and resolve

```bash
git merge --no-commit --no-ff origin/main
```

**Resolution policy (decided 2026-09-21): prefer upstream; port only unique
hardening.** When both sides solved the same problem, take upstream's
implementation and carry forward only what upstream genuinely lacks. This keeps
the fork delta small, which is what keeps the weekly sync cheap. When a
collision is large enough that taking upstream drops merged fork work, surface
it rather than deciding silently.

### Never use `git checkout --ours|--theirs` on a large file

It replaces the **whole file** with that side's version, discarding every
non-conflicting change the other side made. It is only safe when the conflict
hunk covers all of the file's variable content.

- Safe: `prebuilt/*.json` — the hunk spans `markdown`/`mermaid`/`generated_at`,
  and only `section_id`/`title` sit outside it (verified identical).
- **Not safe:** `architecture-facts.yaml` — 4500 lines, one hunk. `--ours`
  there silently discarded 611 upstream insertions; the failure surfaced only
  as two `test_doc_code_consistency.py` failures. Recover with
  `git checkout -m -- <file>` to re-create the conflict, then edit the hunk.

To check before using it:

```bash
git show :1:<file> >/dev/null   # base exists
git diff --name-only $(git merge-base origin/main nemsoft/main) origin/main -- <file>
# then confirm upstream's changes to that file are confined to the hunk
```

### Conflict hotspot map

| File | Collides because | Usual resolution |
|---|---|---|
| `providers/openai_compatible.py` · `registry.py` · `translator.py` · `config.py` · `.env.example` | both sides add settings/provider fields | union both; keep `LOCAL_*` and `SEARXNG_*` intact |
| `orchestrator/executive.py` | both sides rewrite the tool loop | read both sides in full; most of this file auto-merges and is unreviewed |
| `integrations/slack_bot.py` · `api/main.py` | both built Slack-in-lifespan independently | upstream (2026-09-21); fork's `EmbeddedSlackBot` removed |
| `architecture/prebuilt/*.json` · `architecture-facts.yaml` · `docs/architecture.md` | both re-author the same sections | re-author against MERGED code, never pick a side |

### Architecture docs are re-authored, not merged

Neither side's text describes the merged system. Read the merged source, then
write what is true. The 2026-09-21 sync had to scrub every `talent` claim from
`agents.json`, `mcp_server.json` and `architecture-facts.yaml` because upstream
had deleted the specialist — including inside `mermaid` strings, which no
prose-level sweep catches. Validate every edit:

```bash
python3 -m json.tool packages/core/openexecutive/architecture/prebuilt/<id>.json > /dev/null
python3 -c "import yaml;yaml.safe_load(open('packages/core/openexecutive/architecture/architecture-facts.yaml'))"
```

Read a section's markdown with
`python3 -c "import json;print(json.load(open('<path>'))['markdown'])"`.

## Step 5 — sweeps before staging

```bash
# markers, anchored AND loose — a formatter can indent one past the anchored sweep
git grep -nE '^(<<<<<<<|=======|>>>>>>>)' -- .
grep -rnaE '<<<<<<<|>>>>>>>' packages/ docs/ evals/ knowledge/ .env.example
```

`-a` is required: host `grep` is `ugrep` and silently skips files it calls
binary. Then the duplicate-block sweep — when both sides add the same block git
merges both copies cleanly and flags nothing. Each must return exactly 1:

```bash
for p in "def _routing_prepass" "def resolve_specialist_name" "def to_specialist_block" \
         "def select_web_search_tool" "def _build_body" "def _log_bot_crash"; do
  printf '%-34s %s\n' "$p" "$(grep -rac "$p" --include='*.py' packages/core/openexecutive/ | awk -F: '{s+=$2} END{print s+0}')"
done
```

Finally, re-read every "the only …" / "today X is the only …" / "never" claim in
the touched files. Post-merge most of them are false.

## Step 6 — fork-invariant checklist

Work already merged into the fork that a bad resolution would silently drop.
Each is one grep; all must be present, and the named tests must pass.

| Invariant | Where | Test |
|---|---|---|
| `resolve_specialist_name` + `route_parallel(conversation_context=...)` | `orchestrator/router.py` | `test_specialist_name_resolution.py` |
| `_routing_prepass` | `orchestrator/executive.py` | `test_routing_prepass.py` |
| "Consulting Your Leadership Team" section | `prompts/executive_persona.py` | `test_routing_prepass.py` |
| `LOCAL_MODELS` / `LOCAL_BASE_URL` / `LOCAL_REASONING_EFFORT` | `config.py`, `providers/registry.py` | `test_config_local_models.py`, `test_provider_registry.py` |
| `SEARXNG_URL` client-side `web_search` | `config.py`, `orchestrator/searxng_search.py` | `test_searxng_search.py`, `test_client_tool_loop.py` |
| `to_specialist_block` profile digest | `memory/company_profile.py` | `test_executive_conversation_context_wiring.py` |
| `routing_anomaly` attached to its turn | `orchestrator/router.py` | `test_audit_route.py` |

## Step 7 — gate

```bash
cd packages/core
uv sync && uv lock --check                    # upstream bumps deps; CI gates lock freshness
cd .. && make lint                            # ruff + mypy
cd packages/core && env -u BACKEND_SHARED_SECRET -u OE_PUBLIC_DEPLOYMENT uv run pytest tests/unit/ -q
cd .. && toolbox run -c rust-dev npm --prefix packages/ui run build
```

- `BACKEND_SHARED_SECRET` set → TestClient tests return 401. `OE_PUBLIC_DEPLOYMENT`
  set → the app raises at **import**, so it looks like a collection error.
- `npm run lint` has no ESLint config and opens an interactive prompt — `npm run
  build` is the UI's gate. `npm`/`node` exist only inside `toolbox run -c rust-dev`.
- Known-red on main and excluded from comparison:
  `tests/integration/test_chat_committee.py::test_chat_with_committee_streams_phases_and_revised_text`.

**Expect failures here even when the merge was conflict-free** — they are the
semantic conflicts the textual merge could not see. The 2026-09-21 sync had 13,
of which 10 came from one signature change (a fork parameter that upstream's new
tests did not pass) and 1 from a test whose premise upstream had deleted.

## Step 8 — review and PR

Scope code review to the **conflict-resolution surface** — the files you hand-
resolved plus the auto-merged regions of `executive.py`. Running the full
pre-PR chain over an 18k-line upstream import produces noise about code nobody
here wrote; say so explicitly in the PR body so it is not mistaken for a
full-diff review.

PR body is three sections and nothing else (`.github/PULL_REQUEST_TEMPLATE.md`):
**Problem**, **Approach**, **Checklist**. Name the commit range imported, every
silent behaviour change from Step 2, and the baseline-vs-merged test counts.

```bash
git push -u nemsoft chore/sync-upstream-<YYYY-MM-DD>
gh pr create --repo nemsoft-eu/OpenExecutive --base main --draft --body-file <file>
```

`gh` runs on the HOST as plain `gh` — never `toolbox run -c rust-dev gh`, which
reports "not logged into any GitHub hosts". Open as draft, then request
`@bugbot review` and `@codex review` as two separate comments.

**Never merge without the user's explicit authorisation for that specific PR.**
Green CI, clean bots and a plan that says "then merge" are not authorisation.
