import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import glob
import os

nbs_to_skip = ['Flexfringe_Hyperparameter_Study.ipynb']

os.chdir('.')
for notebook_filename in glob.glob("*.ipynb"):

    if notebook_filename in nbs_to_skip:
        continue

    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600)
        ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})

        with open(notebook_filename, 'w+', encoding='utf-8') as f:
            nbformat.write(nb, f)
