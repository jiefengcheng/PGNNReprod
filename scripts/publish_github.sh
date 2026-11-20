#!/usr/bin/env bash
# Source-only publishing helper for the existing PGNNReprod repository.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO='jiefengcheng/PGNNReprod'

if ! command -v git >/dev/null 2>&1; then
  echo 'ERROR: git is required.' >&2; exit 1
fi
remote="$(git remote get-url origin 2>/dev/null || true)"
case "$remote" in
  "https://github.com/$REPO.git"|"https://github.com/$REPO"|"git@github.com:$REPO.git") ;;
  *) echo "ERROR: expected origin to be $REPO, got: $remote" >&2; exit 1 ;;
esac
branch="$(git branch --show-current)"
if [ "$branch" != 'main' ]; then
  echo "ERROR: expected main branch, got: $branch" >&2; exit 1
fi

git add README.md .gitignore .gitattributes .github pyproject.toml notebooks src scripts tests docs
python3 scripts/check_no_data.py
if git diff --cached --quiet; then
  echo 'No staged changes; nothing to publish.'; exit 0
fi
git commit -m 'Update independent TxGNN reproduction source-only workspace'
git push origin main
