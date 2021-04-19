import subprocess
from autohooks.api import ok
from autohooks.api.git import get_staged_status, stash_unstaged_changes

PYTHON_FILES_EXTENSION = ".py"
JUPYTER_NOTEBOOKS_EXTENSION = ".ipynb"


def get_files_from_commit(extention: str) -> list[str]:
    return [f for f in get_staged_status() if str(f).endswith(extention)]


def precommit(**kwargs) -> int:
    def get_valid_filenames(files: list[str]) -> list[str]:
        return [str(file).split()[1] for file in files]

    py_files = get_files_from_commit(PYTHON_FILES_EXTENSION)
    notebooks = get_files_from_commit(JUPYTER_NOTEBOOKS_EXTENSION)
    if py_files:
        with stash_unstaged_changes(py_files):
            filenames = get_valid_filenames(py_files)
            files_str = ", ".join(filenames)
            subprocess.run(["flake8", *filenames], check=True)
            ok(f"Files {files_str} checked with flake8")
            subprocess.run(["black", *filenames], check=True)
            ok(f"Files {files_str} linted with black")
    if notebooks:
        with stash_unstaged_changes(notebooks):
            filenames = get_valid_filenames(notebooks)
            files_str = ",".join(filenames)
            subprocess.run(["flake8-nb", *filenames], check=True)
            ok(f"Notebooks {files_str} checked with flake8-nb")
            subprocess.run(["jblack", *filenames], check=True)
            ok(f"Notebooks {files_str} linted with jblack")
    return 0
