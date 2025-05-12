# MACE_LES_OPENMM Setup Guide

Run MACE-LES using OpenMM with the following steps.

## **Environment Setup**

```bash
export CONDA_OVERRIDE_CUDA=11.8
```

## **Clone Repositories**

```bash
mkdir NEWFILE
cd NEWFILE
git clone https://github.com/ChengUCB/mace.git mace_les
git clone https://github.com/ChengUCB/les.git
curl -o mace-les-openmm.yml https://raw.githubusercontent.com/ChengUCB/openmm-ml/main/mace-les-openmm.yml
```
---


## **Create and Activate Conda Environment**

```bash
mamba env create -f mace-les-openmm.yml
conda activate mace-les-openmm

cd mace_les
pip install -e .
cd ../les
pip install -e .
```

## **Example**
One can obtain the template script for MD simulations. 

```bash
curl -o NPT.py https://raw.githubusercontent.com/ChengUCB/openmm-ml/main/examples/MACE-LES/NPT.py
```

---

# NOTICE

## **Modifications**


