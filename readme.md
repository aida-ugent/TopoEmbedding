![Banner](/Scripts/banner.png)

# Topologically Regularized Data Embeddings
The code in this repository is accompanying the manuscript "Topologically Regularized Data Embeddings".

## Installation

### Python
We provide a [conda_env.yml](conda_env.yml) file listing the required packages. You can install by creating a new conda environment
```bash
conda env create -f topembedding/conda_env.yml -p ./topo_env
conda activate topo_env/
```
#### (1) Install [TopologyLayer](https://github.com/bruel-gabrielsson/TopologyLayer)
```bash
pip install git+https://github.com/bruel-gabrielsson/TopologyLayer.git
```

#### (2) Install [Aleph](https://github.com/Pseudomanifold/Aleph.git) by following the instructions on their GitHub.
```
git clone https://github.com/Pseudomanifold/Aleph.git
cd Aleph && mkdir build && cd build && cmake ../ && make && make test

cd bindings/python/aleph
python setup.py install
```

#### (3) Optional: Install [DioDe](https://github.com/mrzv/diode#diode) to exerun the pseudotime analysis in the [CellCyle notebook](Experiments/CellCycle.ipynb).


### R 
To execute the R scripts in the `Scripts` folder you need:
* TDA
* ggplot2
* latex2exp
* gridExtra
	
## Repository Content
[(Back to top)](#topologically-regularized-data-embeddings)

### Code
Try out the example config files in `/Code/config` by providing one of them to main.py. 
```bash
python main.py Code/config/synthetic_random_optimize.yaml
```
Note that 'cell_cycle.yaml' takes about 5 Minutes to run. Upon completion the final embeddings will be shown.

### Data
* Synthetic data are generated using [Data/datasets.py](Data/datasets.py)
* Cell trajectory [(source)][celltrajectory]: included as [Data/CellCycle.rds](/Data/CellCycle.rds)
* Cell bifurcation [(source)][cellbifurcation]: included as [Data/CellBifurcation.rds](/Data/CellBifurcating.rds)
* Karate[1]: partially (graph) loaded from networkx and partially (weights) from [Data/Karate.txt](/Data/Karate.txt)
* Harry Potter [(source)][harrypotter]: included in [Data/HarryPotter](/Data/HarryPotter/)

### Experiments
In this folder you find one notebook for every dataset to reproduce the results in the experiments section. You can open the .hml version to see the code with output.

### Scripts
Folder "Scripts": contains two Jupyter notebooks and two R scripts to reproduce the visualizations of Section (2) in the manuscript.


## Acknowledgements
[(Back to top)](#topologically-regularized-data-embeddings)

The content of this repository is an extension of the code in [topembedding](https://github.com/robinvndaele/topembedding.git) developed by Robin Vandaele. 


## References
[celltrajectory]: https://zenodo.org/record/1443566/files/real/gold/cell-cycle_buettner.rds?download=1
[cellbifurcation]: https://zenodo.org/record/1443566/files/real/gold/cellbench-SC1_luyitian.rds?download=1
[harrypotter]: https://github.com/hzjken/character-network
[1]: W.W. Zachary. An information flow model for conflict and fission in small groups. Journal of Anthropological Research, 33:452–473, 1977.