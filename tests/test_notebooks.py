import glob
import pytest
import nbformat
import papermill as pm


all_notebooks = [n for n in glob.glob("tutorials/**/*.ipynb", recursive=True)]


@pytest.mark.parametrize("notebook", sorted(all_notebooks))
def test_notebooks(notebook):
    """Test Notebooks in the tutorial root folder."""
    res = pm.execute_notebook(
        notebook,
        "/dev/null",
        parameters={
            "epochs": 1,
            "n_test_batches": 5,
            "n_train_items": 64,
            "n_test_items": 64,
        },
        timeout=300,
    )
    assert isinstance(res, nbformat.notebooknode.NotebookNode)
