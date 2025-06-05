"""
macepotential.py: Implements the MACE potential function.

2025-06-05 Xiaoyu Wang
This is a modified version of the MACE potential function.
To-do:
clean up the code, make it more readable and efficient. 


This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Stephen Farr, Joao Morado

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import openmm # Updated import
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory
from typing import Iterable, Optional, Tuple


class MACEPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates MACEPotentialImpl objects."""

    def createImpl(
        self, name: str, model_path: Optional[str] = None, **args
    ) -> MLPotentialImpl:
        return MACEPotentialImpl(name, model_path)


class MACEPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the MACE potential.

    The MACE potential is constructed using MACE to build a PyTorch model,
    and then integrated into the OpenMM System using a TorchForce.
    This implementation supports both MACE-OFF23 and locally trained MACE models.

    To use one of the pre-trained MACE-OFF23 models, specify the model name. For example:

    >>> potential = openmmml.MLPotential('mace-off23-small') # Corrected example with module

    Other available MACE-OFF23 models include 'mace-off23-medium' and 'mace-off23-large'.

    To use a locally trained MACE model, provide the path to the model file. For example:

    >>> potential = openmmml.MLPotential('mace', model_path='MACE.model') # Corrected example with module

    During system creation, you can optionally specify the precision of the model using the
    ``precision`` keyword argument. Supported options are 'single' and 'double'. For example:

    >>> system = potential.createSystem(topology, precision='single')

    By default, the implementation uses the precision of the loaded MACE model.
    According to the MACE documentation, 'single' precision is recommended for MD (faster but
    less accurate), while 'double' precision is recommended for geometry optimization.

    Additionally, you can request computation of the full atomic energy, including the atom
    self-energy, instead of the default interaction energy, by setting ``returnEnergyType`` to
    'energy'. For example:
    
    >>> system = potential.createSystem(topology, returnEnergyType='energy')

    The default is to compute the interaction energy, which can be made explicit by setting
    ``returnEnergyType='interaction_energy'``.

    Attributes
    ----------
    name : str
        The name of the MACE model.
    model_path : str
        The path to the locally trained MACE model if ``name`` is 'mace'.
    """

    def __init__(self, name: str, model_path: Optional[str]) -> None:
        self.name = name
        self.model_path = model_path

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        precision: Optional[str] = None,
        returnEnergyType: str = "interaction_energy",
        **args,
    ) -> None:
        import torch
        import openmmtorch # Should be openmmtorch, not openmm.torch

        try:
            from mace.tools import utils, to_one_hot, atomic_numbers_to_indices
            from mace.calculators.foundations_models import mace_off
        except ImportError as e:
            raise ImportError(
                f"Failed to import mace with error: {e}. "
                "Install mace with 'pip install mace-torch'."
            )
        try:
            from e3nn.util import jit
        except ImportError as e:
            raise ImportError(
                f"Failed to import e3nn with error: {e}. "
                "Install e3nn with 'pip install e3nn'."
            )
        try:
            from NNPOps.neighbors import getNeighborPairs
        except ImportError as e:
            raise ImportError(
                f"Failed to import NNPOps with error: {e}. "
                "Install NNPOps with 'conda install -c conda-forge nnpops'."
            )

        assert returnEnergyType in [
            "interaction_energy",
            "energy",
        ], f"Unsupported returnEnergyType: '{returnEnergyType}'. Supported options are 'interaction_energy' or 'energy'."

        if self.name.startswith("mace-off23"):
            size = self.name.split("-")[-1]
            assert (
                size in ["small", "medium", "large"]
            ), f"Unsupported MACE model: '{self.name}'. Available MACE-OFF23 models are 'mace-off23-small', 'mace-off23-medium', 'mace-off23-large'"
            model = mace_off(model=size, device="cpu", return_raw_model=True)
        elif self.name == "mace":
            if self.model_path is not None:
                model = torch.load(self.model_path, map_location="cpu")
            else:
                raise ValueError("No model_path provided for local MACE model.")
        else:
            raise ValueError(f"Unsupported MACE model: {self.name}")

        model = jit.compile(model)

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        atomicNumbers = [atom.element.atomic_number for atom in includedAtoms]

        modelDefaultDtype = next(model.parameters()).dtype
        if precision is None:
            dtype = modelDefaultDtype
        elif precision == "single":
            dtype = torch.float32
        elif precision == "double":
            dtype = torch.float64
        else:
            raise ValueError(
                f"Unsupported precision {precision} for the model. "
                "Supported values are 'single' and 'double'."
            )
        if dtype != modelDefaultDtype:
            print(
                f"Model dtype is {modelDefaultDtype} "
                f"and requested dtype is {dtype}. "
                "The model will be converted to the requested dtype."
            )

        zTable = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        nodeAttrs = to_one_hot(
            torch.tensor(
                atomic_numbers_to_indices(atomicNumbers, z_table=zTable),
                dtype=torch.long,
            ).unsqueeze(-1),
            num_classes=len(zTable),
        )

        class MACEForce(torch.nn.Module):
            # Annotate instance variables at the class level for TorchScript
            # if they are not assigned in __init__ from parameters or other complex logic
            # However, most are assigned from parameters or buffers, which TorchScript handles.
            # The `torch.jit.Attribute` wrapper is an alternative for complex types or empty lists/dicts.
            # For simple types assigned from parameters, explicit class-level annotation is often not strictly needed
            # if __init__ signatures are typed, but the warning suggests it for "empty non-base types".
            # Here, all buffers/parameters are directly created, and types are inferable.
            # Let's ensure TorchScript compatibility by being explicit where necessary.
            
            # It seems the warning "The TorchScript type system doesn't support instance-level annotations on empty non-base types in __init__"
            # is general. Given the current structure where buffers are registered, it might be a false positive or apply to other patterns.
            # We will proceed without adding explicit Attribute wrappers unless a specific attribute causes a JIT error.

            def __init__(
                self,
                model: torch.jit._script.RecursiveScriptModule, # Keep type hint specific for clarity
                nodeAttrs: torch.Tensor,
                atoms_indices: Optional[torch.Tensor], # Explicitly torch.Tensor or None
                periodic: bool,
                dtype: torch.dtype,
                returnEnergyType: str,
                z_table_in: utils.AtomicNumberTable, # Changed name to avoid conflict
                r_max_model: float, # Pass r_max from the model
            ) -> None:
                super(MACEForce, self).__init__()

                self.dtype_force = dtype # Renamed to avoid potential conflict if self.dtype existed
                self.model_force = model.to(self.dtype_force) # Renamed
                self.energyScale = 96.4853
                self.lengthScale = 10.0
                self.returnEnergyType_force = returnEnergyType # Renamed

                # Storing atoms_indices directly (already a tensor or None)
                if atoms_indices is not None:
                     self.register_buffer("indices", atoms_indices)
                else:
                    # To make TorchScript happy with potentially unassigned attribute
                    self.indices: Optional[torch.Tensor] = None


                self.register_buffer("ptr", torch.tensor([0, nodeAttrs.shape[0]], dtype=torch.long)) # removed requires_grad=False, default for buffers
                self.register_buffer("node_attrs_force", nodeAttrs.to(self.dtype_force)) # Renamed
                self.register_buffer("batch_force", torch.zeros(nodeAttrs.shape[0], dtype=torch.long)) # Renamed
                self.register_buffer("pbc_force", torch.tensor([periodic, periodic, periodic], dtype=torch.bool)) # Renamed
                
                self.r_max_force = r_max_model # Store r_max

                self.node_e0: Optional[torch.Tensor] = None # Explicit type hint
                if hasattr(self.model_force, 'atomic_energies_fn'):
                    # Ensure node_attrs_force is used here
                    self.node_e0 = self.model_force.atomic_energies_fn(self.node_attrs_force).detach()
                
                if self.returnEnergyType_force == "interaction_energy" and self.node_e0 is None:
                    raise RuntimeError(
                        "Cannot compute 'interaction_energy'. "
                        "The MACE model does not have an 'atomic_energies_fn' or it's not recognized."
                    )

            def _getNeighborPairs(
                self, positions: torch.Tensor, cell: Optional[torch.Tensor]
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                # Using self.r_max_force which was passed from the model
                neighbors, wrappedDeltas, _, _ = getNeighborPairs(
                    positions, self.r_max_force, -1, cell
                )
                mask = neighbors >= 0
                neighbors = neighbors[mask].view(2, -1)
                wrappedDeltas = wrappedDeltas[mask[0], :]

                edgeIndex = torch.hstack((neighbors, neighbors.flip(0))).to(torch.int64)
                if cell is not None:
                    deltas = positions[edgeIndex[0]] - positions[edgeIndex[1]]
                    wrappedDeltas = torch.vstack((wrappedDeltas, -wrappedDeltas))
                    
                    # Directly calculate inverse; TorchScript does not support try-except here.
                    # A singular cell matrix is an issue with the simulation setup.
                    inv_cell = torch.linalg.inv(cell)
                    
                    shiftsIdx = torch.mm(deltas - wrappedDeltas, inv_cell)
                    # Clamp shiftsIdx to avoid large values if numerics are an issue, though typically not needed for valid cells
                    # shiftsIdx = torch.round(shiftsIdx) # Or torch.round() if expecting integer shifts
                    shifts = torch.mm(shiftsIdx, cell)
                else:
                    shifts = torch.zeros((edgeIndex.shape[1], 3), dtype=self.dtype_force, device=positions.device)

                return edgeIndex, shifts

            def forward(
                self, positions: torch.Tensor, boxvectors: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
                current_positions = positions
                if self.indices is not None:
                    current_positions = current_positions[self.indices]

                current_positions = current_positions.to(self.dtype_force) * self.lengthScale

                current_cell: Optional[torch.Tensor] = None
                if boxvectors is not None:
                    current_cell = boxvectors.to(self.dtype_force) * self.lengthScale
                
                edgeIndex, shifts = self._getNeighborPairs(current_positions, current_cell)

                inputDict = {
                    "ptr": self.ptr,
                    "node_attrs": self.node_attrs_force, # Use renamed attribute
                    "batch": self.batch_force, # Use renamed attribute
                    "pbc": self.pbc_force, # Use renamed attribute
                    "positions": current_positions,
                    "edge_index": edgeIndex,
                    "shifts": shifts,
                    "cell": current_cell if current_cell is not None else torch.zeros(3, 3, dtype=self.dtype_force, device=current_positions.device),
                }

                out = self.model_force(inputDict, compute_force=False)

                model_total_energy = out.get("energy")
                
                final_energy: Optional[torch.Tensor] = None # Explicit type hint
                if self.returnEnergyType_force == "interaction_energy":
                    if model_total_energy is not None:
                        if self.node_e0 is None:
                             raise RuntimeError("Cannot compute 'interaction_energy' because node_e0 is not available.")
                        final_energy = model_total_energy - self.node_e0.sum()
                    else:
                        model_node_energy = out.get("node_energy")
                        if model_node_energy is not None:
                            if self.node_e0 is None:
                                raise RuntimeError("Cannot compute 'interaction_energy' from node_energy because node_e0 is not available.")
                            final_energy = (model_node_energy - self.node_e0).sum()
                        else:
                            raise RuntimeError(
                                "Could not compute 'interaction_energy'. "
                                "Model output missing 'energy' or 'node_energy' keys."
                            )
                elif self.returnEnergyType_force == "energy":
                    if model_total_energy is None:
                        raise RuntimeError("Model did not return 'energy' for returnEnergyType='energy'.")
                    final_energy = model_total_energy
                else:
                    raise ValueError(f"Unsupported returnEnergyType: '{self.returnEnergyType_force}'")

                if final_energy is None: # Should not happen if logic above is correct
                     raise AssertionError("Final energy was not computed.")

                return final_energy * self.energyScale

        isPeriodic = (
            topology.getPeriodicBoxVectors() is not None
        ) or system.usesPeriodicBoundaryConditions()

        # Prepare atoms_indices for MACEForce constructor
        atoms_indices_tensor: Optional[torch.Tensor] = None
        if atoms is not None:
            atoms_indices_tensor = torch.tensor(sorted(list(atoms)), dtype=torch.int64)


        maceForce = MACEForce(
            model,
            nodeAttrs,
            atoms_indices_tensor, # Pass the tensor or None
            isPeriodic,
            dtype,
            returnEnergyType,
            zTable, # Pass zTable, though not explicitly used in the MACEForce methods shown, good for consistency
            float(model.r_max.item()) if isinstance(model.r_max, torch.Tensor) else float(model.r_max) # Pass r_max
        )

        module = torch.jit.script(maceForce)

        force = openmmtorch.TorchForce(module)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(isPeriodic)
        system.addForce(force)
