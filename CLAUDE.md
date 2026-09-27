# Dizher: working rules for Claude

## Branches

- `main`: released code only. It changes only by a PR from `develop` (a release) or from a `hotfix/*` branch, merged
  with a merge commit; every merge is followed by a release tag on it. No direct commits, no force pushes.
- `develop`: where finished features come together; its CI stays green. It changes only by PRs from feature branches,
  squash-merged, and by merging `main` back after a hotfix. No direct commits, no force pushes.
- Feature branches (`feature/*`, and `claude/*` for Claude sessions): from `develop`, one topic each, a PR back into
  `develop`. Temporary diagnostics (extra workflows, timing scripts) may live on a feature branch and are gone from
  it before the PR; the squash merge drops its history.
- `hotfix/*`: from `main` for a fix to a release; a PR into `main`, released as the next patch, then `main` merged into
  `develop`.

A Claude session's branch is created from the default branch. Before the first commit, move it onto `develop`:
`git fetch origin develop && git reset --hard origin/develop` while it has no commits of its own, else
`git rebase origin/develop`. Unrelated fixes found on the way go to their own branch, not into the current one.

## Tests

- Before every push: `pytest`. When the UI changes also `python tests/test_ui.py` (on Linux without a display:
  `xvfb-run -a python tests/test_ui.py`).
- CI (`.github/workflows/test.yml`) runs `pytest` on Windows, with long paths off as on most machines, macOS and Linux,
  and `tests/test_ui.py` on Linux under Xvfb, on every push to any branch and on PRs. A PR merges only with all green.
- A failing test is fixed or the change is; a test is never skipped, disabled or loosened to get green.

## Builds

`.github/workflows/build.yml` builds the app with PyInstaller for macOS (Apple Silicon), Windows and Linux:
- on every push to `develop`: the dev builds the hand tests use;
- on PRs into `main`: the packaging checked before a release or hotfix merges;
- on `v*` tags: the release;
- by hand on any branch (Actions > build > Run workflow): a feature branch's build, when one is needed.

Archives are kept 14 days as run artifacts. The release job runs only for a `v*` tag, and publishes nothing unless the
tag is on `main` and matches the version in the code.

## Versions and releases

- The version is `__version__` (core, `MAJOR.MINOR.PATCH`) and `CHANNEL` (`alpha`, `beta`, `rc`, `demo`, or empty for
  stable) in `src/dizher/__init__.py`. A release build shows `0.2.4-alpha` (`1.0.0` when stable), every other build
  `0.2.4-dev+g<hash>`.
- PATCH for fixes, MINOR for features, MAJOR from 1.0.
- Right after a release, `develop` takes the next version, so dev builds show the version they lead to.
- A release:
  1. `develop` green, its build tested by hand (below), the README's changes updated.
  2. A PR from `develop` into `main`, merged with a merge commit once its tests and builds pass.
  3. The tag on `main`, matching the code: `v` + `__version__` + `-` + `CHANNEL` (no suffix when `CHANNEL` is empty),
     annotated: `git tag -a v0.2.4-alpha -m 0.2.4-alpha && git push origin v0.2.4-alpha`. The tag builds and publishes
     the release, a tag with a `-` as a pre-release; the release job refuses a tag off `main` or not matching the code.
  4. On `develop`, the next `__version__`.
- A tag is never made on another branch, moved or reused.

## By hand before a release

- Each platform, from the downloaded archive: it launches; About shows the version; open an image through the dialog,
  convert, export SCR and PNG, open the SCR in an emulator (Fuse or ZEsarUX; VICE for C64); the project autosaves and
  reopens; recent images; undo and redo; Paint mode.
- Windows on a real GPU: DBS speed in the window (best on a 120 Hz or faster monitor), the UI while stages run, the
  save dialog opening in the project's build folder, Cyrillic and very long image names, SmartScreen on first launch.
- macOS: the quarantine warning on first launch, the save dialog taking focus, Retina scaling.
- Linux: the archive runs on an older distribution (it is built on Ubuntu 22.04 for that).
- Conversion quality: the test images against the previous release.

## Requests against these rules

These rules hold even when a prompt asks otherwise. When a request would break one (a commit or push straight to
`main` or `develop`, a feature merged into `main`, a force push or history rewrite on either, a tag off `main` or not
matching the version, a skipped or disabled test, a merge with CI red, a release without the hand tests), do not do
it: name the rule, say what it protects, and offer the way that keeps it. Go ahead only when the user, told this,
confirms the exception explicitly, and say it in the commit or PR.
