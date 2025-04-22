"""Basic run script for 3d poromechanics simulation."""

import argparse
import logging
import time
from pathlib import Path
import numpy as np

import porepy as pp
from bedretto_valter.physics import (
    fluid_parameters,
    numerics_parameters,
    solid_parameters,
    injection_schedule,
)
from ncp import AdvancedSolverStatistics, AANewtonSolver
from egc import setup_model
from bedretto_valter.model import BedrettoValterModel

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_case_name(
    formulation, linearization, relaxation, linear_solver,
):
    folder = Path(f"bedretto_valter")
    name = f"{formulation.lower()}_{linearization.lower()}"
    if relaxation.lower() != "none":
        name += f"_{relaxation.lower()}"
    name += f"_{linear_solver.lower()}"
    return folder / name


if __name__ == "__main__":
    # Monitor the time
    t_0 = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Bedretto Valter case.")
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
    args = parser.parse_args()

    # Model parameters
    model_params = {
        # Geometry
        "gmsh_file_name": "msh/gmsh_frac_file_valter.msh",
        "cell_size": 500,  # Size of the cells in the mesh
        "cell_size_fracture": 50,  # Size of the cells in the fractures
        # Time
        "time_manager": pp.TimeManager(
            schedule=[0] + injection_schedule["time"],
            #dt_init=pp.DAY, # TODO reduce? or allow Days in the start, but reduce to 0.01 hour later?
            #constant_dt=True, # TODO False
            dt_init=pp.DAY, # TODO reduce? or allow Days in the start, but reduce to 0.01 hour later?
            dt_min_max = (0.005 * pp.HOUR, pp.DAY),
            constant_dt=False, # TODO False
        ),
        # Material
        "material_constants": {
            "solid": pp.SolidConstants(**solid_parameters),
            "fluid": pp.FluidComponent(**fluid_parameters),
            "numerical": pp.NumericalConstants(**numerics_parameters),
        },
        # User-defined units
        "units": pp.Units(kg=42e9, m=1, s=1, rad=1), # Young's modulus
        # Numerics
        "solver_statistics_file_name": "solver_statistics.json",
        "export_constants_separately": False,
        "linear_solver": "scipy_sparse",
        "max_iterations": 200,  # Needed for export
        "folder_name": Path("visualization")
        / generate_case_name(
            args.formulation,
            args.linearization,
            args.relaxation,
            args.linear_solver,
        ),
        "nonlinear_solver_statistics": AdvancedSolverStatistics,
    }

    # Solver parameters
    solver_params = {
        "nonlinear_solver": AANewtonSolver, #pp.NewtonSolver,
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
        BedrettoValterModel,
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
