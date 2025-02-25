import subprocess
from pathlib import Path

import pytest

PROJ_ROOT = Path(__file__).parents[1]


def get_notebooks(pattern=None, exclude=None):
    nbpaths = [
        str(p)
        for p in (PROJ_ROOT / "docs" / "examples").glob("*.py")
        if pattern is None or pattern in p.name
    ]

    # sort for pytest-xdist: workers must collect tests in the same order
    return sorted([p for p in nbpaths if not exclude or not any(e in p for e in exclude)])


@pytest.mark.slow
@pytest.mark.example
@pytest.mark.parametrize("notebook", get_notebooks())
def test_notebooks(notebook):
    args = ["jupytext", "--from", "py", "--to", "ipynb", "--execute", notebook, "-k", "xattree"]
    proc = subprocess.run(args, capture_output=True, text=True)
    stdout = proc.stdout
    stderr = proc.stderr
    returncode = proc.returncode
    assert returncode == 0, f"could not run {notebook}:\n{stdout}\n{stderr}"
