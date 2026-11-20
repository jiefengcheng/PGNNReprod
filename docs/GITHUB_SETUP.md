# GitHub workspace

Repository: https://github.com/jiefengcheng/PGNNReprod

This project is committed to the user-created repository. It was observed as
**public** at the initial setup. Change visibility through GitHub Settings if
private access is desired; the code contains no biomedical source datasets.

## Clone and verify

```bash
git clone https://github.com/jiefengcheng/PGNNReprod.git
cd PGNNReprod
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

## Local publishing, after editing

```bash
# In the cloned repository, with your own Git identity/authentication configured:
bash scripts/publish_github.sh
```

The script stages an explicit allowlist, performs the no-data check, commits if
there are changes, and pushes to `origin/main`. It refuses to run outside a clone
whose `origin` URL matches the canonical repository.

Never force-add raw knowledge graph files, arrays, trained weights, checkpoints,
patient-level records, data-bearing output logs, or generated model artifacts.
If such material enters Git history, removing it from the latest commit is not
sufficient; rotate secrets where necessary and rewrite the affected history.
