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
    if 'MACE-OFF' in MODEL_PATH:
        return MODEL_PATH

    # process the MACE-LES model 
    model = torch.load(MODEL_PATH, map_location=DEVICE)

    if hasattr(model, 'les'):
        if hasattr(model.les, 'atomwise') and hasattr(model.les.atomwise, 'outnet') and model.les.atomwise.outnet is None:
            r = torch.rand(10, 3, device=device)
            model.les.atomwise(r, batch=torch.zeros(r.shape[0], dtype=torch.int64, device=device))

        # add the feature 
        model.les.use_atomwise = False
    
    else:
        print("Warning: Model does not have 'les' attribute, skipping configuration")
        
    torch.save(model, './converted_model.pt')            
    return "./converted_model.pt"

# Model configuration
MODEL_PATH = "../" # PATH of the model 
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
FRICTION_COEFF = 1.0              # default: 1 / unit.picosecond; 0.1 for liquids
TIMESTEP = 1.0                    # default: unit.femtosecond 
OUTPUT_DIR = "output_md"          # output directory

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
    max_n_pairs=-1,     # set as -1 to use all pairs
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
