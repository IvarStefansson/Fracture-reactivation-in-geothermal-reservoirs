"""Basic run script for 3d poromechanics simulation."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import porepy as pp
from porepy.numerics.nonlinear import line_search
from simple_bedretto.geometry import BedrettoGeometry
from common.numerics import (
    DarcysLawAd,
    ReverseElasticModuli,
)
from simple_bedretto.physics import (
    Physics,
    fluid_parameters,
    numerics_parameters,
    solid_parameters,
)
from common.contact_mechanics import AuxiliaryContact
from common.fracture_states import FractureStates
from common.iteration_export import IterationExporting
from common.norms import LebesgueConvergenceMetrics
from common.statistics import AdvancedSolverStatistics, LogPerformanceDataVectorial

from common.contact_mechanics import (
    LinearRadialReturnTangentialContact,
    NCPContact,
)

from common.newton_return_map import NewtonReturnMap
from FTHM_Solver.hm_solver import IterativeHMSolver


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
): """Simple Bedretto model solved with Huebers nonlinear radial return formulation."""


class LinearRadialReturnModel(
    LinearRadialReturnTangentialContact, NonlinearRadialReturnModel
): """Simple Bedretto model solved with Alart linear radial return formulation."""

class NCPModel(
    NCPContact,
    NonlinearRadialReturnModel,
): """Simple Bedretto model solved with NCP formulation."""


def generate_case_name(ad_mode, formulation):
    return f"{ad_mode}_{formulation}"


if __name__ == "__main__":
    # Monitor the time
    t_0 = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run simple Bedretto case.")
    parser.add_argument("--ad-mode", type=str, default="picard", help="AD mode.")
    parser.add_argument(
        "--formulation", type=str, default="rr-nonlinear", help="Formulation to use."
    )
    parser.add_argument(
        "--num-fractures", type=int, default=6, help="Number of fractures."
    )
    args = parser.parse_args()
    ad_mode = args.ad_mode
    formulation = args.formulation

    # Model parameters
    model_params = {
        # Geometry
        "gmsh_file_name": f"msh/gmsh_frac_file.msh",
        "num_fractures": args.num_fractures,
        # Time
        "time_manager": pp.TimeManager(
            # TODO allow for negative times in PP for initialization
            schedule=[0, 2 * pp.DAY] + [(3+i) * pp.DAY for i in range(5)],
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
        "folder_name": Path("visualization/simple_bedretto")
        / generate_case_name(ad_mode, formulation),
        "nonlinear_solver_statistics": AdvancedSolverStatistics,
    }
    Path(model_params["folder_name"]).mkdir(parents=True, exist_ok=True)

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

    match ad_mode:
        case "picard":
            ...

        case "newton":

            class NonlinearRadialReturnModel(DarcysLawAd, NonlinearRadialReturnModel): ...

            class LinearRadialReturnModel(
                DarcysLawAd, LinearRadialReturnModel
            ): ...

            class NCPModel(DarcysLawAd, NCPModel): ...

        case _:
            raise ValueError(f"AD mode {ad_mode} not recognized.")

    # Model setup
    logger.info(f"\n\nRunning {model_params['folder_name']}")
    if formulation == "rr-nonlinear":
        model = NonlinearRadialReturnModel(model_params)

    elif formulation == "rr-linear":
        model = LinearRadialReturnModel(model_params)

    elif formulation == "ncp-min":
        model_params["ncp_type"] = "min"
        model = NCPModel(model_params)

    elif formulation == "ncp-fb":
        model_params["ncp_type"] = "fb"
        model = NCPModel(model_params)

    elif formulation == "ncp-fb-full":
        model_params["ncp_type"] = "fb-full"
        model = NCPModel(model_params)

    elif formulation == "rr-nonlinear-linesearch":
        
        class NonlinearRadialReturnModel(
            pp.models.solution_strategy.ContactIndicators,
            NonlinearRadialReturnModel,
        ):
            """Added contact indicators for line search."""
        model = NonlinearRadialReturnModel(model_params)

        class ConstraintLineSearchNonlinearSolver(
            line_search.ConstraintLineSearch,  # The tailoring to contact constraints.
            line_search.SplineInterpolationLineSearch,  # Technical implementation of the actual search along given update direction
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

    elif formulation == "rr-nonlinear-return-map":
        
        class NonlinearRadialReturnModel(
            NewtonReturnMap,
            NonlinearRadialReturnModel,
        ):
            """Add return map before each iteration."""

        model = NonlinearRadialReturnModel(model_params)

    elif formulation == "rr-nonlinear-fthm":

        class NonlinearRadialReturnModel(
            IterativeHMSolver,
            NonlinearRadialReturnModel,
        ):
            """Add return map before each iteration."""
        model = NonlinearRadialReturnModel(model_params)
        model_params["linear_solver_config"] = {
            "solver": "CPR",  # Avaliable options: CPR, SAMG, FGMRES (fastest to slowest).
            "ksp_monitor": False,  # Enable to see convergence messages from PETSc.
            "logging": False,  # Does not work well with a progress bar.
        }
        solver_params["linear_solver_config"] = model_params["linear_solver_config"]


    else:
        raise ValueError(f"formulation {formulation} not recognized.")

    pp.run_time_dependent_model(model, solver_params)

    logger.info(
        f"\nTotal number of iterations: {model.nonlinear_solver_statistics.cache_num_iteration}"
    )
    logger.info(f"\nTotal time: {time.time() - t_0:.2f} s")
