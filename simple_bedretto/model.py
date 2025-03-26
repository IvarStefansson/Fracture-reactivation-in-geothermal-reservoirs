"""Basic run script for 3d poromechanics simulation."""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import porepy as pp
from icecream import ic
from setups.geometry import BedrettoGeometry
from setups.numerics import AdaptiveCnum, DarcysLawAd, MinFbSwitch, ReverseElasticModuli
from setups.physics import (
    ExtendedNumericalConstants,
    Physics,
    fluid_parameters,
    injection_schedule,
    numerics_parameters,
    solid_parameters,
)
from setups.statistics import AdvancedSolverStatistics, LogPerformanceDataVectorial

import ncp

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ScaledRadialReturnModel(
    ReverseElasticModuli,  # Characteristic displacement from traction
    BedrettoGeometry,  # Geometry
    Physics,  # BC and IC
    ncp.AuxiliaryContact,  # Yield function, orthognality, and alignment
    ncp.FractureStates,  # Physics based conact states
    ncp.IterationExporting,  # Tailored export
    ncp.LebesgueConvergenceMetrics,  # Convergence metrics
    LogPerformanceDataVectorial,  # Tailored convergence checks
    pp.constitutive_laws.CubicLawPermeability,  # Basic constitutive law
    pp.poromechanics.Poromechanics,  # Basic model
):
    """Mixed-dimensional poroelastic problem."""


class ScaledLinearRadialReturnModel(
    ncp.LinearRadialReturnTangentialContact, ScaledRadialReturnModel
): ...


# class UnscaledRadialReturnModel(
#     AdaptiveCnum,
#     UnscaledContact,
#     ScaledRadialReturnModel,
# ): ...


# NCP Formulations
class ScaledNCPModel(
    AdaptiveCnum,
    # MinFbSwitch,
    ncp.ScaledContact,
    ncp.NCPNormalContact,
    ncp.NCPTangentialContact2d,
    ScaledRadialReturnModel,
): ...


# NCP Formulations
class NCPModel(ncp.UnscaledContact, ScaledNCPModel): ...


def generate_case_name(ad_mode, mode, cnum, tol, unitary_units):
    return f"{ad_mode}_{mode}_cnum_{cnum}_tol_{tol}_unitary_units_{unitary_units}"


if __name__ == "__main__":
    # Monitor the time
    t_0 = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run single fracture test cases.")
    parser.add_argument(
        "--num_time_steps", type=int, default=5, help="Number of time steps."
    )
    parser.add_argument(
        "--num_iter", type=int, default=200, help="Number of nonlinear iterations."
    )
    parser.add_argument("--ad-mode", type=str, default="picard", help="AD mode.")
    parser.add_argument("--mode", type=str, default="ncp-min", help="Method to use.")
    parser.add_argument(
        "--linear-solver", type=str, default="scipy_sparse", help="Linear solver."
    )
    parser.add_argument("--tol", type=float, default=1e-10, help="Tolerance.")
    parser.add_argument("--cnum", type=float, default=1e0, help="Cnum")
    parser.add_argument("--regularization", type=str, default="none")
    parser.add_argument("--unitary_units", type=str, default="True", help="Units.")
    parser.add_argument("--mesh_size", type=float, default=10, help="Mesh size.")
    parser.add_argument(
        "--output", type=str, default="visualization", help="base output folder"
    )
    args = parser.parse_args()

    num_time_steps = args.num_time_steps
    num_iter = args.num_iter
    ad_mode = args.ad_mode
    mode = args.mode
    linear_solver = args.linear_solver
    cnum = args.cnum
    tol = args.tol
    mesh_size = args.mesh_size
    if np.isclose(mesh_size, int(mesh_size)):
        mesh_size = int(mesh_size)
    regularization = args.regularization
    unitary_units = args.unitary_units == "True"

    # Model parameters
    model_params = {
        # Geometry
        "cell_size_fracture": args.mesh_size,
        "gmsh_file_name": f"msh/gmsh_frac_file_mesh_{mesh_size}.msh",
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
            # "numerical": pp.NumericalConstants(**numerics_parameters), # NOTE: Use tailored open state tol
        },
        "units": (
            pp.Units(kg=1e0, m=1e0, s=1, rad=1)
            if unitary_units
            else pp.Units(kg=1e10, m=1, s=1, rad=1)
        ),
        # Numerics
        "stick_slip_regularization": regularization,
        "solver_statistics_file_name": "solver_statistics.json",
        "export_constants_separately": False,
        "linear_solver": linear_solver,  # Needed for setting up solver
        "max_iterations": num_iter,  # Needed for export
        "folder_name": Path(args.output)
        / f"mesh_size_{mesh_size}"
        / generate_case_name(ad_mode, mode, cnum, tol, unitary_units),
        "nonlinear_solver_statistics": AdvancedSolverStatistics,
    }
    Path(model_params["folder_name"]).mkdir(parents=True, exist_ok=True)

    # Physics-based tuning/scaling of numerical parameters
    # Use open state tolerance model parameters according to user input
    characteristic_contact_traction = (
        injection_schedule["reference_pressure"]
        if mode in ["rr-nonlinear", "rr-linear", "ncp-min-scaled"]
        else 1.0
    )
    open_state_tolerance = (
        tol
        if mode in ["rr-nonlinear", "rr-linear", "ncp-min-scaled"]
        else tol * injection_schedule["reference_pressure"]
    )
    numerics_parameters.update(
        {
            "open_state_tolerance": open_state_tolerance,
            "contact_mechanics_scaling": cnum,
            "contact_mechanics_scaling_t": cnum,
            "characteristic_contact_traction": characteristic_contact_traction,
        }
    )

    model_params["material_constants"]["numerical"] = ExtendedNumericalConstants(
        **numerics_parameters
    )

    # Solver parameters
    solver_params = {
        "nonlinear_solver": ncp.AANewtonSolver,
        "max_iterations": num_iter,
        "aa_depth": 0,
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
    ic(model_params["folder_name"])
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

    elif mode == "ncp-min-alternative-stick":
        model_params["ncp_type"] = "min-alternative-stick"
        model = NCPModel(model_params)

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
        raise ValueError(f"Mode {mode} not recognized. Choose 'ncp' or 'rr'.")

    pp.run_time_dependent_model(model, solver_params)

    logger.info(
        f"\nTotal number of iterations: {model.nonlinear_solver_statistics.cache_num_iteration}"
    )
    logger.info(f"\nTotal time: {time.time() - t_0:.2f} s")
