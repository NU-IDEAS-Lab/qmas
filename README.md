# QMAS Project

This repository contains the policy/training code for the QMAS Project.

Code from this repository has been used for the following publications:
* AAMAS 2027 submission: [89a19d9f5dedf2032d01967d6e81e410d624eba6](https://github.com/NU-IDEAS-Lab/qmas/tree/89a19d9f5dedf2032d01967d6e81e410d624eba6)

## Package Description
Packages are as follows:

 * **onpolicy**: Contains the algorithm code.
 * **flocking_zoo**: Contains the Flocking environment code. We probably won't use this but it's a decent reference.
 * **patrolling_zoo**: Contains the Patrolling environment code. We can use this as a simpler scenario than the SMACv2 scenario.

## Installation

 1) Clone the qmas_task_allocation repository:
    ```bash
    git clone --recurse git@github.com:NU-IDEAS-Lab/qmas_task_allocation.git
    ```

 2) Create a Conda environment with required packages:
    ```bash
    cd ./qmas_task_allocation
    conda env create -n qmas -f ./environment.yml
    conda activate qmas
    ```

 3) Install the local packages (`onpolicy`, etc.) in development mode:
    ```
    python -m pip install -e .
    ```
