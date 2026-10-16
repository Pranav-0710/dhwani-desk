# STRUCTURE.md

## Repo layout

```
dhwani-desk/
├── backend/            # FastAPI: audio proxy, Gnani clients, dialogue, Laya serving, tickets
│   ├── app.py          # entry point, WebSocket routes
│   ├── gnani/          # Prisma stream client, Timbre client (thin interfaces)
│   ├── audio/          # resample to 8 kHz, noise injection, barge-in
│   ├── dialogue/       # state machine + Kannada reply templates
│   ├── decisions/      # Laya schema, router, thresholds
│   ├── tickets/        # SQLite models, masking, retention
│   └── tests/
├── console/            # React + TS + Vite: phone UI + supervisor console
├── data/               # synthetic data generators, seed + test sets (synthetic only)
├── training/           # Kaggle / Colab notebooks (outputs stripped)
├── eval/               # run.py, reports
├── docs/               # PRD.md, CONTRIBUTIONS.md, design notes
├── .github/            # CI, PR template, issue template
├── AGENTS.md           # rules for coding agents
└── docker-compose.yml
```

Folders are created by the first PR that needs them; CI jobs skip until their folder exists.

Owners: Jey Pranav (A) — `backend/`, phone UI in `console/`. Lakshmi Narasimman (B) — `data/`, `training/`, `eval/`, supervisor console in `console/`. `docs/` is shared.

## Branch naming

`type/issue-number-short-description`

Types match commit types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`.

Examples: `feat/7-prisma-proxy`, `fix/21-barge-in-race`.

Use the real issue number from GitHub — check it before naming the branch.

## Commit message format

`type: short description` — under 72 characters. No vague messages. No co-author trailers.

## Workflow: issue → merged PR

```
Before starting:
1. Check GitHub Issues — is the task already assigned?
2. Assign yourself to the issue
3. git switch main && git pull origin main
4. git switch -c [type]/[issue-number]-[short-description]

While working:
5. git add [specific files] — never git add .
6. git commit -m "type: description"
7. git merge main (resolve conflicts locally first)
8. git push -u origin [branch]

Finishing:
9.  Open a PR — fill in the template; title in `type: description` form
10. Link the issue with "Closes #[number]"
11. The other person is requested as reviewer automatically (`.github/CODEOWNERS`)
12. Wait for CI to pass
13. Get approval → Squash and merge → Delete branch

After merge:
14. git switch main && git pull origin main
15. git branch -d [your-branch]
16. Close the issue if it didn't auto-close
```

## Review

- Every PR needs 1 approval from the other person: Pranav reviews Lakshmi's PRs, Lakshmi reviews Pranav's
- Changes to the Laya decision schema, escalation thresholds or the ticket model: both review together
- The reviewer runs the change locally when it touches the call path

## Branch protection (repo owner sets once)

Settings → Rules → Rulesets → New branch ruleset, target `main`:

- Restrict deletions; block force pushes; require linear history
- Require a pull request before merging: 1 approval, dismiss stale approvals on new commits, require conversation resolution
- Require status checks to pass: `PR hygiene`, `Secret scan`, `Backend`, `Console`, `Notebooks`; require branches to be up to date
- Allowed merge method: squash only (Settings → General → Pull Requests)

## CI

| Job | Checks |
| --- | --- |
| PR hygiene | Title `type: description` ≤ 72 chars, body contains `Closes #N`, no co-author trailers in PR commits |
| Secret scan | gitleaks over the full history |
| Backend | `uv sync --locked`, `ruff check`, `ruff format --check`, `pytest` (skips until `backend/pyproject.toml` exists) |
| Console | `npm ci`, `lint`, `typecheck`, `test`, `build` scripts (skips until `console/package.json` exists) |
| Notebooks | Every committed `.ipynb` has no outputs or execution counts |

`backend/pyproject.toml` must list `ruff` and `pytest` as dev dependencies. `console/package.json` must define `lint`, `typecheck`, `test` (non-watch) and `build` scripts.

## Hugging Face Hub

Datasets and checkpoints never go in git. They live in private repos under the Hugging Face organization **Dhwanidesk**:

| Repo | Type | Holds | Written by | Read by |
| --- | --- | --- | --- | --- |
| `Dhwanidesk/dhwani-desk-data` | dataset | generated training set: `train.jsonl`, `rejected.jsonl`, `report.json`, dataset card | `training/02_generate_data.ipynb` (#7, #12) | fine-tuning notebook (#14) |
| `Dhwanidesk/dhwani-desk-laya` | model | fine-tuned Laya checkpoints | fine-tuning notebook (#14) | backend, at `LAYA_CHECKPOINT_REVISION` (#15) |

Every version used for training or serving gets a tag (for example `data-v1`, `laya-v1`), and the backend pins a tag or commit, never `main`.

**Tokens**
- Kaggle / Colab: a **write** token stored as the notebook secret `HF_TOKEN` (Kaggle: Add-ons → Secrets; Colab: the key icon). Never in a cell.
- Laptop: a **read** token in `.env` as `HF_TOKEN`. A fine-grained token limited to the Dhwanidesk repos is best.
- Never commit a token, print it, or paste it into chat, issues or PRs.

**Access**
- Members of the Dhwanidesk organization see the private repos: Jey Pranav (admin) and Lakshmi Narasimman (write). Members are managed on the organization page → Settings → Members.
- Each member uses their own tokens: a read token in their own `.env`, and a write token in their own Kaggle / Colab secrets.
- The repos stay private. The upload code refuses to push to a repo that is public.

