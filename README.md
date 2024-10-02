# Flocking Zoo

This repository contains the policy/training code for the Adversarial Sheep Project.

## Package Description
Packages are as follows:

 * **onpolicy**: Contains the algorithm code.
 * **flocking_zoo**: Contains the environment code.

## Installation

 1) Clone the flocking_zoo repository:
    ```bash
    git clone --recurse git@github.com:NU-IDEAS-Lab/flocking_zoo.git
    ```

 2) Create a Conda environment with required packages:
    ```bash
    cd ./flocking_zoo
    conda env create -n flocking_zoo -f ./environment.yml
    conda activate flocking_zoo
    ```

 3) Install PyTorch to the new `flocking_zoo` conda environment using the [steps outlined on the PyTorch website](https://pytorch.org/get-started/locally/).

 4) Install the `onpolicy` and `flocking_zoo` packages:
    ```
    python -m pip install -e .
    ```

## Operation

You may run the example in `onpolicy/scripts/train_flocking_scripts/mappo.ipynb`.