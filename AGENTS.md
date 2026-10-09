# AGENTS.md — Instructions for coding agents

Every coding agent working in this repo follows these rules, whatever tool it is. Humans follow them too.

## Project context

Dhwani Desk is a Kannada-first voice helpline for civic complaints, built for the Great Indian AI Internship Challenge (deadline 2026-11-10). Gnani Prisma v2.5 transcribes the caller, Laya (open-source decision engine) makes every routing and escalation decision locally, and Gnani Timbre v2.5 speaks the reply. Source of truth for product and architecture: `docs/PRD.md`.

Stack:

- Backend: Python 3.11, FastAPI, WebSockets, SQLite (SQLModel), managed with `uv`, linted with `ruff`
- Decisions: Laya multilingual checkpoint, ONNX on CPU; fine-tuned checkpoint pulled from a private Hugging Face Hub repo
- Console + phone UI: React + TypeScript + Vite, managed with `npm`
- Training and data generation: Kaggle / Google Colab notebooks only — never on a laptop

Directories and owners:

- `backend/` — audio proxy, Gnani clients, dialogue state machine, Laya serving, tickets. Owner: Jey Pranav (A)
- `console/` — phone UI (A) and supervisor console (B)
- `data/` — synthetic data generators, seed and test sets. Owner: Lakshmi Narasimman (B)
- `training/` — Kaggle / Colab notebooks. Owner: B
- `eval/` — evaluation harness and reports. Owner: B
- `docs/` — PRD, contributions log, design notes. Shared

## Git workflow

### Branch

- Confirm the current branch before starting: `git branch --show-current`
- Work only on the branch named in the task handoff; never switch branches mid-task without instruction
- Branch names: `type/issue-number-short-description` (e.g. `feat/7-prisma-proxy`)

### Commits

- Format: `type: short description`, under 72 characters
- Types: feat, fix, chore, docs, refactor, test
- No vague messages ("updated files", "misc changes", "fixed stuff")
- Stage only files relevant to the task — never `git add .` or `git add -A`
- No co-author trailers and no tool attribution of any kind in commits, PR text, issues, comments or files. Commits carry only the human author

### Hard rules

- Never push to `main`; never merge PRs; never force-push a shared branch
- Commit or push only to the assigned branch, and only when the human working on that branch allows it
- Never close or edit issues, or comment on PRs, on someone's behalf without their confirmation
- Never create or modify `.github/workflows/` without explicit instruction
- If CI fails on your branch, report the failure and wait for instruction
- When done, list every file modified or created

## File scope

- Only modify files directly related to the task, inside the scope given in the handoff
- Never modify without explicit instruction: `AGENTS.md`, `.github/`, `backend/pyproject.toml`, `backend/uv.lock`, `console/package.json`, `console/package-lock.json`, `docs/PRD.md`
- Shared contract — flag to both owners before changing: the Laya decision schema (`backend/decisions/schema.py`), escalation thresholds, and the ticket model. The console, the state machine and the eval harness all depend on them
- Never delete files without explicit instruction

## Code conventions

- Python: `ruff check` and `ruff format` clean; type hints on public functions; tests in `backend/tests/` with `pytest`
- TypeScript: ESLint clean, `tsc --noEmit` clean; components PascalCase, files kebab-case
- Import order: external packages, then internal modules, then relative imports
- No commented-out code; no leftover `print` / `console.log` debugging
- Every external AI component (Prisma, Timbre, Laya) sits behind a thin internal interface in `backend/gnani/` or `backend/decisions/` — application logic never calls a vendor SDK or endpoint directly
- No LLM in the live call path. Replies come from templates; decisions come from Laya
- Every Laya answer that drives behaviour is stored with its probability and the checkpoint revision

## Notebooks

- Notebooks live in `training/` and run on Kaggle or Colab
- Commit them with outputs and execution counts stripped (`pipx run nbstripout training/*.ipynb`); CI rejects notebooks with outputs
- Read tokens from Kaggle / Colab secrets, never from a cell literal

## Environment and secrets

- Never hardcode secrets, API keys or tokens; never print them in logs
- Variables go in `.env` (never committed); `.env.example` lists every required variable
- Never read or output the contents of `.env` files
- Synthetic data only: no real phone numbers, Aadhaar, PAN, names of real people or recorded real calls
- The demo city is fictional ("Nagara Seva, City Corporation of Hosapura"); never name or imitate a real authority

## Task handoff format

Every task starts with this header. Confirm each field before starting work:

```
Issue: #[number] — [title]
Branch: [branch-name]
Scope: [files/directories you may touch]
Do not touch: [files/directories that are off limits]

Acceptance criteria:
- [ ] [criterion 1]
- [ ] [criterion 2]

Task description:
[the actual task]

When done:
- List every file modified
- List every file created
- Flag any shared files that need changes but were not in scope
```
