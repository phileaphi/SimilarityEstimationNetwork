#!/bin/bash
if [ -z "$(conda env list | grep automl)" ]; then
    # Install environment from scratch
    conda env create -f .environment.yml
else
    # Install changes according to .yml file
    conda env update -f .environment.yml --prune
fi
conda activate automl