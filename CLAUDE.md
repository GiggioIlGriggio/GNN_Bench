# CLAUDE.md

This file is read by Claude Code. It documents agent skill configuration for this repo.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (use the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Git workflow

Repo lives on GitHub (`GiggioIlGriggio/GNN_Bench`). Claude commits after every meaningful change. Features go in their own `git worktree` + `feature/<slug>` branch, merged back to `main` via PR. See `docs/agents/git-workflow.md`.
