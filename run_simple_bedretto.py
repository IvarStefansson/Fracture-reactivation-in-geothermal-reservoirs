"""Basic run script for 3d poromechanics simulation."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import porepy as pp
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
    ScaledContact,
    NCPNormalContact,
    NCPTangentialContact,
    UnscaledContact,
)


# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ScaledRadialReturnModel(
    ReverseElasticModuli,  # Characteristic displacement from traction
    BedrettoGeometry,  # Geometry
    Physics,  # BC and IC
    AuxiliaryContact,  # Yield function, orthognality, and alignment
    FractureStates,  # Physics based conact states
    IterationExporting,  # Tailored export
    LebesgueConvergenceMetrics,  # Convergence metrics
    LogPerformanceDataVectorial,  # Tailored convergence checks
    pp.constitutive_laws.CubicLawPermeability,  # Basic constitutive law
    pp.poromechanics.Poromechanics,  # Basic model
):
    """Mixed-dimensional poroelastic problem."""


class ScaledLinearRadialReturnModel(
    LinearRadialReturnTangentialContact, ScaledRadialReturnModel
): ...


# NCP Formulations
class ScaledNCPModel(
    ScaledContact,
    NCPNormalContact,
    NCPTangentialContact,
    ScaledRadialReturnModel,
): ...


# NCP Formulations
class NCPModel(UnscaledContact, ScaledNCPModel): ...


def generate_case_name(ad_mode, mode):
    return f"{ad_mode}_{mode}"


if __name__ == "__main__":
    # Monitor the time
    t_0 = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run simple Bedretto case.")
    parser.add_argument("--ad-mode", type=str, default="picard", help="AD mode.")
    parser.add_argument(
        "--mode", type=str, default="rr-nonlinear", help="Formulation to use."
    )
    args = parser.parse_args()
    ad_mode = args.ad_mode
    mode = args.mode

    # Model parameters
    model_params = {
        # Geometry
        "gmsh_file_name": f"msh/gmsh_frac_file.msh",
        # Time
        "time_manager": pp.TimeManager(
            schedule=[0, 5 * pp.DAY],
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
        / generate_case_name(ad_mode, mode),
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

            class ScaledNCPModel(DarcysLawAd, ScaledNCPModel): ...

            class NCPModel(DarcysLawAd, NCPModel): ...

            class ScaledRadialReturnModel(DarcysLawAd, ScaledRadialReturnModel): ...

            class ScaledLinearRadialReturnModel(
                DarcysLawAd, ScaledLinearRadialReturnModel
            ): ...

        case _:
            raise ValueError(f"AD mode {ad_mode} not recognized.")

    # Model setup
    logger.info(f"\n\nRunning {model_params['folder_name']}")
    if mode == "rr-nonlinear":
        model = ScaledRadialReturnModel(model_params)

    elif mode == "rr-linear":
        model = ScaledLinearRadialReturnModel(model_params)

    elif mode == "ncp-min":
        model_params["ncp_type"] = "min"
        model = NCPModel(model_params)

    elif mode == "ncp-min-scaled":
        model_params["ncp_type"] = "min"
        model = ScaledNCPModel(model_params)

    elif mode == "ncp-fb":
        model_params["ncp_type"] = "fb"
        model = NCPModel(model_params)

    elif mode == "ncp-fb-full":
        model_params["ncp_type"] = "fb-full"
        model = NCPModel(model_params)

    # elif mode == "rr-linesearch":
    # Need to integrate pp.models.solution_strategy.ContactIndicators in model class
    #    # porepy-main-1.10

    #    model_params["material_constants"]["solid"]._constants[
    #        "characteristic_displacement"
    #    ] = 1e-2
    #    model = ScaledRadialReturnModel(model_params)

    #    class ConstraintLineSearchNonlinearSolver(
    #        line_search.ConstraintLineSearch,  # The tailoring to contact constraints.
    #        line_search.SplineInterpolationLineSearch,  # Technical implementation of the actual search along given update direction
    #        line_search.LineSearchNewtonSolver,  # General line search.
    #    ): ...

    #    solver_params["nonlinear_solver"] = ConstraintLineSearchNonlinearSolver
    #    solver_params["Global_line_search"] = (
    #        0  # Set to 1 to use turn on a residual-based line search
    #    )
    #    solver_params["Local_line_search"] = (
    #        1  # Set to 0 to use turn off the tailored line search
    #    )
    #    solver_params["adaptive_indicator_scaling"] = (
    #        1  # Scale the indicator adaptively to increase robustness
    #    )

    else:
        raise ValueError(f"Mode {mode} not recognized.")

    pp.run_time_dependent_model(model, solver_params)

    logger.info(
        f"\nTotal number of iterations: {model.nonlinear_solver_statistics.cache_num_iteration}"
    )
    logger.info(f"\nTotal time: {time.time() - t_0:.2f} s")
