import subprocess
from autohooks.api import ok, error
from autohooks.api.git import get_staged_status, stage_files_from_status_list, stash_unstaged_changes
from autohooks.api.path import match
from typing import NoReturn

python_files_extension = "*.py"
jupyter_notebooks_extension = "*.ipynb" 

def get_files_from_commit(extention: str) -> List[str]:
    return [f for f in get_staged_status() if match(f.path, extention)]

def format_files(files: List[str], extension:str) -> NoReturn:
    if not files:
        return
    
    with stash_unstaged_changes(files):
        for file in files:
            if extension == python_files_extension:
                subprocess.run(["flake8", str(file)], check = True)
                ok("File {} formatted successfully".format(str(file)))
            if extension == jupyter_notebooks_extension:
                subprocess.run(["flake8-nb", str(file)], check = True)
                ok("Notebook {} formatted successfully".format(str(file)))
        return    

def precommit(**kwargs):
    python_files = get_files_from_commit(python_files_extension)
    json_files = get_files_from_commit(jupyter_notebooks_extension)

    
