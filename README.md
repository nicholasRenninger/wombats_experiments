# The `wombats` Library

This repo houses the example usage notebooks (`experiments`) as well as the core `wombats` and `flexfringe` libraries powering the framework. 

## To run the experiments

* install `anaconda`
* install `flexfringe` dependencies for you OS, listed [here](https://bitbucket.org/chrshmmmr/dfasat/src/master/).
* clone this repo with: `git clone --recurse-submodules https://github.com/nicholasRenninger/wombats_experiments .`
* change into this repo's directory: `cd wombats_experiments`
* build the `flexfringe` tool: `cd dfasat && make && cd ..`
* create the `conda` environment for this library: `conda env create -f environment.yml`
* activate the conda environment: `conda activate wombats`
* Launch a jupyter server: `jupyter notebook`
* In the jupyter UI, navigate to the `experiments` directory. Each directory has a self-contained experiment which is housed in the `ipynb` (jupyter notebook) file. Open this notebook, and then click "Cell > Run all" and enjoy!


MUCH more documentation to come...
