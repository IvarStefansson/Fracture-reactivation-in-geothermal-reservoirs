"""Basic run script for 3d poromechanics simulation."""

import argparse
import logging
import time
from pathlib import Path
import numpy as np

import porepy as pp
from simple_bedretto.physics import (
    fluid_parameters,
    numerics_parameters,
    solid_parameters,
    injection_schedule,
)
from ncp import AdvancedSolverStatistics
from egc import setup_model
from simple_bedretto.model import SimpleBedrettoTunnel_Model

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_case_name(
    num_fractures, formulation, linearization, relaxation, linear_solver, args_mass_unit
):
    folder = Path(f"simple_bedretto_{num_fractures}")
    name = f"{formulation.lower()}_{linearization.lower()}"
    if relaxation.lower() != "none":
        name += f"_{relaxation.lower()}"
    name += f"_{linear_solver.lower()}"
    if not np.isclose(args_mass_unit, 1e10):
        name += f"_{args_mass_unit}"
    return folder / name


if __name__ == "__main__":
    # Monitor the time
    t_0 = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run simple Bedretto case.")
    parser.add_argument(
        "--formulation",
        type=str,
        default="rr-nonlinear",
        help="""Nonlinear formulation to use (rr-nonlinear [default], """
        """rr-linear, ncp-min, ncp-fb).""",
    )
    parser.add_argument(
        "--linearization",
        type=str,
        default="picard",
        help="AD mode (Picard [default], Newton).",
    )
    parser.add_argument(
        "--relaxation",
        type=str,
        default="None",
        help="Relaxation method (None [default], Picard, Newton).",
    )
    parser.add_argument(
        "--linear-solver",
        type=str,
        default="scipy_sparse",
        help="Linear solver to use. (scipy_sparse [default], pypardiso, fthm).",
    )
    parser.add_argument(
        "--mass-unit",
        type=float,
        default=1e10,
        help="Mass unit (1e10 [default]).",
    )
    parser.add_argument(
        "--num-fractures",
        type=int,
        default=6,
        help="Number of fractures (1-6 [default]).",
    )
    args = parser.parse_args()

    # Model parameters
    model_params = {
        # Geometry
        "gmsh_file_name": "msh/gmsh_frac_file.msh",
        "num_fractures": args.num_fractures,
        "cell_size": 1000,  # Size of the cells in the mesh
        "cell_size_fracture": 500,  # Size of the cells in the fractures
        # Time
        "time_manager": pp.TimeManager(
            schedule=[0] + injection_schedule["time"],
            dt_init=pp.DAY, # TODO reduce? or allow Days in the start, but reduce to 0.01 hour later?
            constant_dt=True, # TODO False
        ),
        # Material
        "material_constants": {
            "solid": pp.SolidConstants(**solid_parameters),
            "fluid": pp.FluidComponent(**fluid_parameters),
            "numerical": pp.NumericalConstants(**numerics_parameters),
        },
        # User-defined units
        "units": pp.Units(kg=args.mass_unit, m=1, s=1, rad=1),
        # Numerics
        "solver_statistics_file_name": "solver_statistics.json",
        "export_constants_separately": False,
        "linear_solver": "scipy_sparse",
        "max_iterations": 200,  # Needed for export
        "folder_name": Path("visualization")
        / generate_case_name(
            args.num_fractures,
            args.formulation,
            args.linearization,
            args.relaxation,
            args.linear_solver,
            args.mass_unit,
        ),
        "nonlinear_solver_statistics": AdvancedSolverStatistics,
    }

    # Solver parameters
    solver_params = {
        "nonlinear_solver": pp.NewtonSolver,
        "max_iterations": 200,
        "nl_convergence_tol": 1e-6,
        "nl_convergence_tol_rel": 1e-6,
        "nl_convergence_tol_res": 1e-6,
        "nl_convergence_tol_res_rel": 1e-6,
        "nl_convergence_tol_tight": 1e-10,
        "nl_convergence_tol_rel_tight": 1e-10,
        "nl_convergence_tol_res_tight": 1e-10,
        "nl_convergence_tol_res_rel_tight": 1e-10,
    }

    # Model setup
    Path(model_params["folder_name"]).mkdir(parents=True, exist_ok=True)
    logger.info(f"\n\nRunning {model_params['folder_name']}")

    (Model, model_params, solver_params) = setup_model(
        SimpleBedrettoTunnel_Model,
        model_params,
        solver_params,
        args.formulation,
        args.linearization,
        args.relaxation,
        args.linear_solver,
    )
    
    # Run the model
    model = Model(model_params)
    pp.run_time_dependent_model(model, solver_params)

    # Simple statistics
    logger.info(
        f"\nTotal number of iterations: {model.nonlinear_solver_statistics.cache_num_iteration}"
    )
    logger.info(f"\nTotal time: {time.time() - t_0:.2f} s")
