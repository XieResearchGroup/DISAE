# DISAE Experiments Replication
These are the steps in replicating DISAE Experiments using DGX-Lei workstation or the Hunter DLS Server. Please follow these steps after cloning the repository and downloading the data. Instructions in doing so are in the README.md file, Steps 2 and 3.

## ----------- INSTRUCTION -----------
### 1. Run a Docker Containter 
Example:
```
nvidia-docker run --name pytorch -it --network=host --rm -v /raid/home/username/:/username nvcr.io/nvidia/pytorch:19.07-py3
```

### 2. Create an RDKit environment
```
conda create -n rdkit-env -c rdkit rdkit libboost=1.65.1
conda activate rdkit-env # If this line does not work, use 'source activate rdkit-env' instead 
```

### 3. Install packages
```
conda install -c conda-forge dataclasses scipy networkx tensorflow scikit-learn
```

### 4. Install version 2.3.0 transformers using pip 
```
pip install transformers==2.3.0
```

### 5. (Optional) Install pytorch and torchvision according to CUDA Version of GPU
```
nvcc --version # Check CUDA Version
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch # Replace 10.1 with appropriate version
```

### 6. View README.md and follow Step 4


