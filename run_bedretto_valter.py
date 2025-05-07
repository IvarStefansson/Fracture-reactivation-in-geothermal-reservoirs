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
)
from ncp import AdvancedSolverStatistics, AdaptiveNewtonSolver
from egc import (
    setup_model,
    AlternatingDecouplingInTime,
    AlternatingDecouplingInNewton,
    SafeNewtonReturnMap,
)
from bedretto_valter.model import BedrettoValterModel

from bedretto_valter.injection import (
    InjectionInterval8,
    WellInjectionInterval8,
    InjectionInterval9,
    WellInjectionInterval9,
    InjectionInterval13,
)

from bedretto_valter.geometry import SmallDisks, IntermediateDisks

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_case_name(
    intervals,
    injection_interval,
    disks,
    dilation,
    formulation,
    linearization,
    relaxation,
    linear_solver,
    mesh_refinement,
    simple_flow,
    tpfa_flow,
    decoupling,
    iterative_decoupling,
    safe_nrm,
    safe_aa,
    safe_relaxation,
):
    folder = Path(f"bedretto_valter")
    geometry = f"int_{intervals}_inj_{injection_interval}_disks_{disks}_dil_{dilation}"
    name = f"{formulation.lower()}_{linearization.lower()}"
    if relaxation.lower() != "none":
        name += f"_{relaxation.lower()}"
    name += f"_mesh{mesh_refinement}"
    name += f"_{linear_solver.lower()}"
    if simple_flow:
        name += "_simple"
    if tpfa_flow:
        name += "_tpfa"
    if decoupling:
        name += "_decoupling"
    if iterative_decoupling:
        name += "_iterative_decoupling"
    if safe_nrm:
        name += "_safe_nrm"
    if safe_aa:
        name += "_safe_aa"
    if safe_relaxation:
        name += "_safe_relaxation"
    return folder / geometry / name


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
        default="linesearch",
        help="Relaxation method (None [default], Picard, Newton).",
    )
    parser.add_argument(
        "--linear-solver",
        type=str,
        default="pypardiso",
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
        "--safe-nrm",
        action="store_true",
        help="If provided, simple flow laws are used",
    )
    parser.add_argument(
        "--safe-aa",
        action="store_true",
        help="If provided, simple flow laws are used",
    )
    parser.add_argument(
        "--safe-relaxation",
        action="store_true",
        help="If provided, simple flow laws are used",
    )
    parser.add_argument(
        "--output", type=str, default="visualization", help="Output folder."
    )
    # Ask for a list of intervals
    parser.add_argument(
        "--intervals",
        type=int,
        nargs="+",
        default=[7, 8, 9, 10, 11, 12, 13, 14],
        help="List of intervals to use for the simulation.",
    )
    # Injection interval
    parser.add_argument(
        "--injection-interval",
        type=str,
        default="8",
        help="Injection interval to use for the simulation.",
    )
    # Small disks
    parser.add_argument(
        "--disks",
        type=str,
        default="large",
        help="Disk size (large [default], small, intermediate).",
    )
    # Dilation angle
    parser.add_argument(
        "--dilation",
        type=float,
        default=0.05,
        help="Dilation angle to use for the simulation.",
    )

    args = parser.parse_args()

    # Mesh refinement
    cell_size = {
        0: 1000,
        1: 100,
        2: 100,
        3: 100,
        4: 100,
        5: 20,
    }
    cell_size_fracture = {
        0: 100,
        1: 50,
        2: 10,
        3: 5,
        4: 2,
        5: 1,
    }

    if args.injection_interval == "8":

        class Injection(InjectionInterval8): ...
    elif args.injection_interval == "9":

        class Injection(InjectionInterval9): ...
    elif args.injection_interval == 13:

        class Injection(InjectionInterval13): ...
    elif args.injection_interval == "8w":

        class Injection(WellInjectionInterval8): ...

    elif args.injection_interval == "9w":

        class Injection(WellInjectionInterval9): ...
    else:
        raise ValueError(
            f"Injection interval {args.injection_interval} not supported. "
            "Please use 8, 9 or 13."
        )

    solid_parameters["dilation_angle"] = args.dilation

    # Model parameters
    model_params = {
        # Numerical modeling/linearization
        "use_simple_flow": args.simple_flow,
        "use_tpfa_flow": args.tpfa_flow,
        # Geometry
        "gmsh_file_name": f"msh/gmsh_frac_file_valter_{args.intervals}_{args.mesh_refinement}.msh",
        "active_intervals": args.intervals,
        "injection_interval": args.injection_interval,
        "cell_size": cell_size[args.mesh_refinement],  # Size of the cells in the mesh
        "cell_size_fracture": cell_size_fracture[
            args.mesh_refinement
        ],  # Size of the cells in the fractures
        # "fracture_network_tolerance": 0.5,
        # Time
        "time_manager": pp.TimeManager(
            schedule=[t for t, _ in Injection().schedule],
            dt_init=0.1 * pp.HOUR,
            dt_min_max=(10 * pp.SECOND, 0.1 * pp.HOUR),
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
            args.intervals,
            args.injection_interval,
            args.disks,
            args.dilation,
            args.formulation,
            args.linearization,
            args.relaxation,
            args.linear_solver,
            args.mesh_refinement,
            args.simple_flow,
            args.tpfa_flow,
            args.decoupling,
            args.iterative_decoupling,
            args.safe_nrm,
            args.safe_aa,
            args.safe_relaxation,
        ),
        "nonlinear_solver_statistics": AdvancedSolverStatistics,
    }

    # Solver parameters
    solver_params = {
        "nonlinear_solver": AdaptiveNewtonSolver,
        "aa_depth": 0,  # Standard Newton, but stops upon cycling
        "max_iterations": 50,
        "nl_convergence_tol": 1e-5,
        "nl_convergence_tol_rel": 1e-5,
        "nl_convergence_tol_res": 1e-5,
        "nl_convergence_tol_res_rel": 1e-5,
        "nl_convergence_tol_tight": 1e-8,
        "nl_convergence_tol_rel_tight": 1e-8,
        "nl_convergence_tol_res_tight": 1e-8,
        "nl_convergence_tol_res_rel_tight": 1e-8,
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

    if args.disks == "large":
        ...
    elif args.disks == "small":

        class Model(SmallDisks, Model): ...
    elif args.disks == "intermediate":

        class Model(IntermediateDisks, Model): ...
    else:
        raise ValueError(
            f"Disk size {args.disks} not supported. "
            "Please use large, small or intermediate."
        )

    if args.safe_nrm:

        class Model(SafeNewtonReturnMap, Model): ...

        model_params["aa_depth"] = -10000  # Use a value not detected by the solver

    if args.safe_aa:
        model_params["aa_depth"] = -1  # Apply AA(1) upon cycling

    if args.safe_relaxation:
        model_params["aa_depth"] = -2  # Apply random relaxation upon cycling

    # Add injection schedule
    class Model(Injection, Model): ...

    # Experimental: loose and iterative coupling
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
