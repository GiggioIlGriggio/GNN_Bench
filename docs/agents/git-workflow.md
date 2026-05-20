# Git workflow

This repo lives on GitHub at `GiggioIlGriggio/GNN_Bench`. Default branch is `main`. Use the `git` and `gh` CLIs for all operations.

## Commit duty

**Claude commits after every meaningful change.** A "meaningful change" is any edit that leaves the repo in a coherent state — a working code change, a doc update, a config tweak. Don't batch unrelated edits into one commit.

- Stage explicitly by filename — never `git add -A` / `git add .` (would sweep up untracked junk).
- One concern per commit. If you touched two unrelated things, make two commits.
- Commit messages follow the imperative-mood convention used by the existing log (`add X`, `fix Y`, `refactor Z`). Subject ≤ 72 chars; body explains *why*, not *what*.
- Include the `Co-Authored-By: Claude` trailer on commits Claude authors.
- **Never** commit `--no-verify`, `--no-gpg-sign`, or `--amend` an already-pushed commit unless the user explicitly asks.
- Never commit secrets — if `.env*` or credentials show up in `git status`, stop and flag it.

If the user gives an instruction that produces a change, the default is: make the change → run any obvious checks → commit. Do not ask "should I commit?" each time; commit unless the user said otherwise for that turn.

## Feature workflow: one git worktree per feature

`main` stays clean. Every feature lives in its own branch *and* its own worktree directory, so multiple in-flight features don't collide.

### Starting a feature

```bash
# from the main checkout (this directory)
git worktree add ../GNN_Bench-<slug> -b feature/<slug>
cd ../GNN_Bench-<slug>
```

- `<slug>` is short, kebab-case, descriptive (`feature/braingnn-pooling`, `feature/wandb-sweep-cleanup`).
- The worktree shares the same `.git` dir as the main checkout — branches, refs, and stashes are visible from both.
- The `.venv/` is gitignored, so each worktree needs its own venv (or symlink the existing one).

### Working in the feature worktree

Same commit duty as above. Push the branch when you have something worth sharing:

```bash
git push -u origin feature/<slug>
```

### Merging back into main

Prefer a PR via `gh` so the merge is reviewable and traceable:

```bash
gh pr create --base main --head feature/<slug> --title "..." --body "..."
# after review / CI
gh pr merge <number> --squash --delete-branch
```

For solo / quick merges, a direct merge is fine:

```bash
cd <main checkout>
git checkout main
git pull
git merge --no-ff feature/<slug>   # --no-ff preserves the feature history as a merge commit
git push
```

`--no-ff` is the default here: it keeps the feature visible in the graph instead of fast-forwarding it into a flat line.

### Cleaning up

```bash
git worktree remove ../GNN_Bench-<slug>
git branch -d feature/<slug>          # delete local branch (use -D if it wasn't merged via PR squash)
git push origin --delete feature/<slug>   # delete remote branch (skipped if gh pr merge --delete-branch did it)
```

### Listing worktrees

```bash
git worktree list
```

## When to break the rules

- **Tiny doc-only edits on `main`** (typo, README fix) — fine to commit directly to `main`. Use judgment: if it could break anything, branch.
- **Hotfixes** — branch off `main` as `hotfix/<slug>`, same worktree flow.
- **WIP commits** — allowed inside a feature branch (mark `WIP:` in the subject). Squash on merge.
