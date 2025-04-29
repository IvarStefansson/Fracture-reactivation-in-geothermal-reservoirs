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
from egc import setup_model, AlternatingDecouplingInTime, AlternatingDecouplingInNewton
from bedretto_valter.model import BedrettoValterModel

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_case_name(
    formulation,
    linearization,
    relaxation,
    linear_solver,
    mesh_refinement,
    simple_flow,
    tpfa_flow,
    decoupling,
    iterative_decoupling,
):
    folder = Path(f"bedretto_valter")
    name = f"{formulation.lower()}_{linearization.lower()}"
    if relaxation.lower() != "none":
        name += f"_{relaxation.lower()}"
    name += f"_mesh{mesh_refinement}"
    name += f"_{linear_solver.lower()}"
    if simple_flow:
        name += "_simple_flow"
    if tpfa_flow:
        name += "_tpfa_flow"
    if decoupling:
        name += "_decoupling"
    if iterative_decoupling:
        name += "_iterative_decoupling"
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
    parser.add_argument(
        "--mesh-refinement",
        type=int,
        default=0,
        help="Mesh refinement level (0 [default], 1, 2).",
    )
    parser.add_argument(
        "--simple_flow",
        action="store_true",
        help="If provided, simple flow laws are used",
    )
    parser.add_argument(
        "--tpfa_flow",
        action="store_true",
        help="If provided, simple flow laws are used",
    )
    parser.add_argument(
        "--decoupling",
        action="store_true",
        help="If provided, alernating decoupling is used",
    )
    parser.add_argument(
        "--iterative-decoupling",
        action="store_true",
        help="If provided, alernating decoupling is used",
    )
    parser.add_argument(
        "--output", type=str, default="visualization", help="Output folder."
    )
    args = parser.parse_args()

    # Mesh refinement
    cell_size = {
        0: 1000,
        1: 100,
        2: 100,
        3: 100,
    }
    cell_size_fracture = {
        0: 100,
        1: 50,
        2: 10,
        3: 1,
    }

    # Model parameters
    model_params = {
        # Numerical modeling/linearization
        "use_simple_flow": args.simple_flow,
        "use_tpfa_flow": args.tpfa_flow,
        # Geometry
        "gmsh_file_name": "msh/gmsh_frac_file_valter.msh",
        "cell_size": cell_size[args.mesh_refinement],  # Size of the cells in the mesh
        "cell_size_fracture": cell_size_fracture[args.mesh_refinement],  # Size of the cells in the fractures
        # Time
        "time_manager": pp.TimeManager(
            schedule=[0] + injection_schedule["time"],
            dt_init=0.125 * pp.DAY,
            dt_min_max=(10 * pp.SECOND, pp.DAY),
            constant_dt=False,
            print_info=True,
            iter_optimal_range=(4, 20),
            iter_max=100,
        ),
        # Material
        "material_constants": {
            "solid": pp.SolidConstants(**solid_parameters),
            "fluid": pp.FluidComponent(**fluid_parameters),
            "numerical": pp.NumericalConstants(**numerics_parameters),
        },
        # User-defined units
        "units": pp.Units(kg=42e9, m=1, s=1, rad=1),  # Young's modulus
        # Numerics
        "solver_statistics_file_name": "solver_statistics.json",
        "linear_solver": "pypardiso",
        "max_iterations": 200,  # Needed for export
        "folder_name": Path(args.output)
        / generate_case_name(
            args.formulation,
            args.linearization,
            args.relaxation,
            args.linear_solver,
            args.mesh_refinement,
            args.simple_flow,
            args.tpfa_flow,
            args.decoupling,
            args.iterative_decoupling,
        ),
        "nonlinear_solver_statistics": AdvancedSolverStatistics,
    }

    # Solver parameters
    solver_params = {
        "nonlinear_solver": AANewtonSolver,  # pp.NewtonSolver,
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

    if args.decoupling:
        class Model(AlternatingDecouplingInTime, Model): ...

    elif args.iterative_decoupling:
        class Model(AlternatingDecouplingInNewton, Model): ...

    # Run the model
    model = Model(model_params)
    pp.run_time_dependent_model(model, solver_params)

    # Simple statistics
    logger.info(
        f"\nTotal number of iterations: {model.nonlinear_solver_statistics.cache_num_iteration}"
    )
    logger.info(f"\nTotal time: {time.time() - t_0:.2f} s")
