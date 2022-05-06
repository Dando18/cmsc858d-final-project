#!/bin/bash

# settings
VENV="./venv"
EXEC="python main.py"

# parameters
DATASETS="mouse-exon"


source ${VENV}/bin/activate

# tsne data
ALGORITHM="tsne"
for dataset in ${DATASETS}; do
    ${EXEC} -d ${dataset} -a ${ALGORITHM}
done
