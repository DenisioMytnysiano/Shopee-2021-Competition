import subprocess
from autohooks.api import ok
from autohooks.api.git import get_staged_status, stash_unstaged_changes
from typing import List

python_files_extension = ".py"
jupyter_notebooks_extension = ".ipynb"


def get_files_from_commit(extention: str) -> List[str]:
    return [f for f in get_staged_status()]


def precommit(**kwargs):
    files = get_files_from_commit(python_files_extension)
    if not files:
        return 0

    with stash_unstaged_changes(files):
        for file in files:
            filename = str(file).split()[1]
            if filename.endswith(python_files_extension):
                subprocess.run(["flake8", filename], check=True)
                ok("File {} formatted with flake8".format(filename))
                subprocess.run(["black", filename], check=True)
                ok("File {} linted with black".format(filename))
            elif filename.endswith(jupyter_notebooks_extension):
                subprocess.run(["flake8-nb", filename], check=True)
                ok("Notebook {} formatted with flake8".format(filename))
                subprocess.run(["jblack", filename], check=True)
                ok("Notebook {} linted with black".format(filename))
    return 0
