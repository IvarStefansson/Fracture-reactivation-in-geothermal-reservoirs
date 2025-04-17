"""Basic run script for 3d poromechanics simulation."""

import argparse
import logging
import time
from pathlib import Path

import porepy as pp
from bedretto_mb1_2d.geometry import BedrettoMB1_Geometry as BedrettoGeometry
from bedretto_mb1_2d.physics import (
    BedrettoMB1_Physics as Physics,
    solid_parameters,
    fluid_parameters,
    numerics_parameters,
)
from ncp import (
    DarcysLawAd,
    ReverseElasticModuli,
    AuxiliaryContact,
    FractureStates,
    IterationExporting,
    LebesgueConvergenceMetrics,
    AdvancedSolverStatistics,
    LogPerformanceDataVectorial,
    LinearRadialReturnTangentialContact,
    NCPNormalContact,
    NCPTangentialContact,
    ScaledContact,
)
from common.newton_return_map import NewtonReturnMap
from FTHM_Solver.hm_solver import IterativeHMSolver
from porepy.numerics.nonlinear import line_search


# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NonlinearRadialReturnModel(
    BedrettoGeometry,  # Geometry
    AuxiliaryContact,  # Yield function, orthognality, and alignment
    FractureStates,  # Physics based contact states for output only
    IterationExporting,  # Tailored export
    LebesgueConvergenceMetrics,  # Convergence metrics
    LogPerformanceDataVectorial,  # Tailored convergence checks
    ReverseElasticModuli,  # Characteristic displacement from traction
    Physics,  # Basic model, BC and IC
):
    """Simple Bedretto model solved with Huebers nonlinear radial return formulation."""


class LinearRadialReturnModel(
    LinearRadialReturnTangentialContact, NonlinearRadialReturnModel
):
    """Simple Bedretto model solved with Alart linear radial return formulation."""


class NCPModel(
    ScaledContact,
    NCPNormalContact,  # Normal contact model
    NCPTangentialContact,  # Tangential contact model
    NonlinearRadialReturnModel,
):
    """Simple Bedretto model solved with NCP formulation."""


def generate_case_name(
    num_fractures, formulation, linearization, relaxation, linear_solver
):
    folder = Path(f"bedretto_mb1_{num_fractures}")
    name = f"{formulation.lower()}_{linearization.lower()}"
    if relaxation.lower() != "none":
        name += f"_{relaxation.lower()}"
    name += f"_{linear_solver.lower()}"
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
        "--num-fractures",
        type=int,
        default=24,
        help="Number of fractures (1-24 [default]).",
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
            schedule=[0, 2 * pp.DAY] + [(3 + i) * pp.DAY for i in range(5)],
            dt_init=pp.DAY,
            constant_dt=True,
        ),
        # Material
        "material_constants": {
            "solid": pp.SolidConstants(**solid_parameters),
            "fluid": pp.FluidComponent(**fluid_parameters),
            "numerical": pp.NumericalConstants(**numerics_parameters),
        },
        # User-defined units
        "units": pp.Units(kg=1e10, m=1, s=1, rad=1),
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

    # Define formulation
    match args.formulation.lower():
        case "rr-nonlinear":
            Model = NonlinearRadialReturnModel

        case "rr-linear":
            Model = LinearRadialReturnModel

        case "ncp-min":
            model_params["ncp_type"] = "min"
            model_params["stick_slip_regularization"] = "origin_and_stick_slip_transition"

            Model = NCPModel

        case "ncp-fb":
            model_params["ncp_type"] = "fb"
            model_params["stick_slip_regularization"] = "origin_and_stick_slip_transition"
            Model = NCPModel

        case "ncp-fb-partial":
            model_params["ncp_type"] = "fb-partial"
            model_params["stick_slip_regularization"] = "origin_and_stick_slip_transition"
            Model = NCPModel

        case _:
            raise ValueError(f"formulation {args.formulation} not recognized.")

    # Choose nonlinear solver (Newton with relaxation)
    match args.linearization.lower():
        case "picard":
            ...

        case "newton":

            class Model(DarcysLawAd, Model):
                """Enhance with AD of permeability."""

        case _:
            raise ValueError(f"AD mode {args.linearization} not recognized.")

    # Choose relaxation method
    match args.relaxation.lower():
        case "none":
            ...

        case "linesearch":

            class Model(
                pp.models.solution_strategy.ContactIndicators,
                Model,
            ):
                """Added contact indicators for line search."""

            class ConstraintLineSearchNonlinearSolver(
                line_search.ConstraintLineSearch,  # The tailoring to contact constraints.
                line_search.SplineInterpolationLineSearch,  # Technical implementation of
                # the actual search along given update direction
                line_search.LineSearchNewtonSolver,  # General line search.
            ): ...

            solver_params["nonlinear_solver"] = ConstraintLineSearchNonlinearSolver
            solver_params["Global_line_search"] = (
                0  # Set to 1 to use turn on a residual-based line search
            )
            solver_params["Local_line_search"] = (
                1  # Set to 0 to use turn off the tailored line search
            )
            solver_params["adaptive_indicator_scaling"] = (
                1  # Scale the indicator adaptively to increase robustness
            )

        case "return-map":

            class Model(
                NewtonReturnMap,
                Model,
            ):
                """Add return map before each iteration."""

        case _:
            raise ValueError(f"Relaxation method {args.relaxation} not recognized.")

    # Choose linear solver
    match args.linear_solver.lower():
        case "scipy_sparse":
            # Use scipy sparse solver
            model_params["linear_solver"] = "scipy_sparse"
            solver_params["linear_solver"] = "scipy_sparse"
        case "pypardiso":
            # Use pypardiso solver
            model_params["linear_solver"] = "pypardiso"
            solver_params["linear_solver"] = "pypardiso"
        case "fthm":

            class Model(
                IterativeHMSolver,
                Model,
            ): ...

            model_params["linear_solver_config"] = {
                # GMRES parameters
                "ksp_atol": 1e-15,
                "ksp_rtol": 1e-10,
                "ksp_max_it": 90,
                "ksp_gmres_restart": 90,
                # Avaliable options for THM: CPR, SAMG, FGMRES (fastest to slowest).
                # For HM, this parameter is ignored.
                "solver": "CPR",
                "ksp_monitor": True,  # Enable to see convergence messages from PETSc.
                "logging": False,  # Does not work well with a progress bar.
                "treat_singularity_contact": True,
            }
            solver_params["linear_solver_config"] = model_params["linear_solver_config"]

        case _:
            raise ValueError(f"Linear solver {args.linear_solver} not recognized.")

    # Run the model
    model = Model(model_params)
    pp.run_time_dependent_model(model, solver_params)

    # Simple statistics
    logger.info(
        f"\nTotal number of iterations: {model.nonlinear_solver_statistics.cache_num_iteration}"
    )
    logger.info(f"\nTotal time: {time.time() - t_0:.2f} s")
