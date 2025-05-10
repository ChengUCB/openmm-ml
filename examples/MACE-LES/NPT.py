from openmmtools.openmm_torch.hybrid_md import PureSystem
import torch
from simtk.openmm import LangevinIntegrator
from openmm import NoseHooverIntegrator, unit
import os
import mace
import les
from e3nn.util import jit

def load_model():
    """Load and configure the MACE model for MD simulation"""
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    
    if 'MACE-OFF' in MODEL_PATH:
        return MODEL_PATH
    
    print("Model architecture:", model.les.atomwise.outnet)
    print("Atomwise layer:", model.les.atomwise)
    
    model_for_md = model.to(DEVICE)
    
    if hasattr(model.les, 'atomwise') and model.les.atomwise.outnet is None:
        r = torch.rand(10, 3, device=DEVICE)
        model.les.atomwise(r, batch=torch.zeros(r.shape[0], dtype=torch.int64, device=DEVICE))
    
    model_for_md.les.use_atomwise = False
    torch.save(model_for_md, './convected_model.pt')
    return "./convected_model.pt"

# Model configuration
MODEL_PATH = "../"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and get path
MLP_path = load_model()

########################################################################################
# Input parameters
INPUT_FILE = "liquid-64.xyz"
TEMPERATURE = 300 * unit.kelvin
PRESSURE = 1.0 * unit.bar # MonteCarloBarostat http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.MonteCarloBarostat.html
thermostat_name = "langevin" # "nose-hoover"
# LangevinIntegrator http://docs.openmm.org/7.1.0/api-python/generated/simtk.openmm.openmm.LangevinIntegrator.html  


# Simulation parameters
FRICTION_COEFF = 1.0    # 1 / unit.picosecond 
TIMESTEP = 1.0          #  unit.femtosecond 
OUTPUT_DIR = "output_md" # output directory

SIM_STEPS = 1000
DUMP_INTERVAL = 100

OUTPUT_FILE = "water_md.pdb"
RESTART = False


# Initialize system
system = PureSystem(
    ml_mol=INPUT_FILE,
    model_path=MLP_path,
    potential="mace", 
    output_dir=OUTPUT_DIR,
    temperature=TEMPERATURE,
    pressure=PRESSURE,
    friction_coeff=FRICTION_COEFF,
    timestep=TIMESTEP,
    nl="torch",         
    max_n_pairs=64,     # set as -1 to use all pairs
    minimiser="openmm"  # or "ase"
) 

# Run molecular dynamics simulation
system.run_mixed_md(
    steps=SIM_STEPS,
    interval=DUMP_INTERVAL,
    output_file=OUTPUT_FILE,
    restart=RESTART,
    integrator_name=thermostat_name
)
