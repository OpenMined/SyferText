import glob
import pytest
import nbformat
import papermill as pm


"""selecting all the notebook in the tutorial folder."""
all_notebooks = [n for n in glob.glob("tutorials/*.ipynb", recursive=True)]

@pytest.mark.parametrize("notebook", sorted(all_notebooks))
def test_notebooks(notebook):
    """Test Notebooks in the tutorial root folder."""
    res = pm.execute_notebook(notebook, "/dev/null", parameters={}, timeout=300)
    assert isinstance(res, nbformat.notebooknode.NotebookNode)
