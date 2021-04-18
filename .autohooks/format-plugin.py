import subprocess
from autohooks.api import ok, error
from autohooks.api.git import get_staged_status, stage_files_from_status_list, stash_unstaged_changes
from autohooks.api.path import match
from typing import NoReturn, List

python_files_extension = "*.py"
jupyter_notebooks_extension = "*.ipynb" 

def get_files_from_commit(extention: str) -> List[str]:
    return [f for f in get_staged_status() if match(f.path, extention)]

def precommit(**kwargs):
    files = get_files_from_commit(python_files_extension)
    if not files:
        return 0
    
    with stash_unstaged_changes(files):
        for file in files:
            subprocess.run(["flake8", str(file)], check = True)
            ok("File {} formatted successfully".format(str(file)))
    return 0   


    
