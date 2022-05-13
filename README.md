# CMSC 858D Final Project

This repo contains the source code for the CMSC 858D final project of Daniel Nichols and Josh Davis.
The scripts implement many experiments for single-cell RNA-seq dimensionality reduction.
See below for how to set up and run the code.

- [CMSC 858D Final Project](#cmsc-858d-final-project)
  - [Install](#install)
  - [Data sets](#data-sets)
    - [mouse-exon](#mouse-exon)
    - [ca1-neurons](#ca1-neurons)
    - [pollen](#pollen)
  - [Running](#running)
  - [Repo Overview](#repo-overview)

## Install
First, clone this repo and run `git submodule update --init --recursive` to clone the submodules.
Then, you need to install the Python dependencies.
To install these run

```bash
# create virtual environment
python3 -m venv venv
source ./venv/bin/activate

pip install -r requirements.txt
```

Now you need to install the SCVis package.
This can be done with the utility script `install-scvis.bash`.

```bash
bash install-scvis.bash
```

_Note:_ if you want to run SCVis you need tensorflow <2.0 which requires Python 3.7 or less. If you want to try with tensorflow>=2.0 then you can uncomment the `tf_upgrade_v2` line in `install-scvis.bash`, but I've had varied results with this.


Next you need to build the densne binary.
This is simply

```bash
cd densvis/densne
g++ sptree.cpp densne.cpp densne_main.cpp -o den_sne -O2
cd ../..
```


## Data sets
The data sets are rather large and, thus, not included in this repo.
We use 3 data sets _mouse-exon_, _ca1-neurons_, and _pollen_.
The instructions for obtaining this data is presented below.

### mouse-exon
The data is available [here](http://celltypes.brain-map.org/rnaseq).
Direct links to zip files are [here](http://celltypes.brain-map.org/api/v2/well_known_file_download/694413985) and [here](http://celltypes.brain-map.org/api/v2/well_known_file_download/694413179).

You will also need `sample_heatmap_plot_data.csv` from [this interactive browser](http://celltypes.brain-map.org/rnaseq/mouse/v1-alm). Click "Sample Heatmaps" -> "Build Plot" -> "Download as CSV".

All of these files should be placed directly in the `./data` subdirectory. 

### ca1-neurons
Download the files from [here](https://figshare.com/articles/dataset/Transcriptomic_analysis_of_CA1_inhibitory_interneurons/6198656) and place the contents in `./data/ca1-neurons`.

### pollen
Download [these two files](https://github.com/hhcho/netsne/tree/master/example_data) from the netSNE repository and place them in `./data/pollen`.


## Running
Most examples can be run as 

```bash
python main.py -d <dataset> -a <algorithm>
```

where _dataset_ is _mouse-exon_, _ca1-neurons_, or _pollen_ and _algorithm_ is _pca_, _lda_, _tsne_, _umap_, _densne_, _densmap_, or _scvis_.
This will run the given algorithm on the given dataset and output a CSV line with the recorded performance metrics.
Additionally, a figure with the computed embedding is dumped into the `./figures` subdirectory.
The full options from `python main.py --help` are

```bash
usage: main.py [-h] [--log {INFO,DEBUG,WARNING,ERROR,CRITICAL}] [-d {synthetic,pollen,mouse-exon,ca1-neurons}] [-a {pca,lda,tsne,umap,hsne,densne,densmap,scvis,netsne}]
               [-p PARAMS] [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --log {INFO,DEBUG,WARNING,ERROR,CRITICAL}
                        logging level
  -d {synthetic,pollen,mouse-exon,ca1-neurons}, --dataset {synthetic,pollen,mouse-exon,ca1-neurons}
  -a {pca,lda,tsne,umap,hsne,densne,densmap,scvis,netsne}, --algorithm {pca,lda,tsne,umap,hsne,densne,densmap,scvis,netsne}
  -p PARAMS, --params PARAMS
                        parameters json file
  -o OUTPUT, --output OUTPUT
                        output csv location
```


_Note:_ Processing the data sets (particularly _mouse-exon_ and _ca1-neurons_) is quite slow. For this reason we checkpoint the intermediate preprocessed data in a pickle file. The first run for a given data set will be slow, but it should speed up after that.


## Repo Overview

The two main files of interest are:

- `dim_reduce.py`: implementations or API calls for all the dimensionality reduction algorithms
- `metrics.py`: performance metrics used to evaluate embeddings