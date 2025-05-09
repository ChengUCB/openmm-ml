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

## **Code Updates Needed**

!!! warning "Important Update Required (2025-05-09)"
    You need to update the MACE code in order to run MD with OpenMM.
    
    **To-do:** Update the MACE branch accordingly.

- **`mace_les/mace/modules/models.py` (around line 669):**

```python
les_energy_opt = les_result['E_lr']
if les_energy_opt is None:
    les_energy = torch.zeros_like(total_energy)
else:
    les_energy = les_energy_opt
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


