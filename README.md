# The `wombats` Library

This repo houses the example usage notebooks (`experiments`) as well as the core `wombats` and `flexfringe` libraries powering the framework. 

## To run the experiments

* install [`anaconda`](https://www.anaconda.com/products/individual)

* install `flexfringe` dependencies for you OS, listed [here](https://bitbucket.org/chrshmmmr/dfasat/src/master/).

* clone this repo with:
 ```bash
git clone --recurse-submodules https://github.com/nicholasRenninger/wombats_experiments .
 ```

* change into this repo's directory:
 ```bash
cd wombats_experiments
 ```
 
* build the `flexfringe` tool:
 ```bash
cd dfasat && make && cd ..
 ```
 
* create the `conda` environment for this library:
```bash
conda env create -f environment.yml
 ```
 
* activate the conda environment:
 ```bash
conda activate wombats
 ```
 
* Launch a jupyter server:
 ```bash
jupyter notebook
 ```
 
* In the jupyter UI, navigate to the `experiments` directory. Each directory has a self-contained experiment which is housed in the `ipynb` (jupyter notebook) file. For example, to see a large example that exercises most of the library, open `experiments/seshia_paper_reproduction/seshia_paper_reproduction.ipynb`. After opening any notebook, click "Cell > Run all" and enjoy!


MUCH more documentation to come...
