"""General helper for setting up a solver strategy."""

from typing import Literal
import porepy as pp
from ncp import (
    DarcysLawAd,
    LinearRadialReturnTangentialContact,
    NCPNormalContact,
    NCPTangentialContact,
    ScaledContact,
)
from porepy.numerics.nonlinear import line_search
from FTHM_Solver.hm_solver import IterativeHMSolver
from egc import NewtonReturnMap
from ncp import NCPContactIndicators

def setup_model(BaseModel, model_params: dict, 
                solver_params: dict,
                formulation: Literal["rr-nonlinear", "rr-linear", "ncp-min", "ncp-fb"] = "rr-nonlinear",
                linearization: Literal["picard", "newton"] = "picard",
                relaxation: Literal["none", "linesearch", "return-map"] = "none",
                linear_solver: Literal["scipy_sparse", "pypardiso", "fthm"] = "scipy_sparse",
                **kwargs):

    # Start with base model
    Model = BaseModel

    # Adapt contact mechanics formulation
    match formulation.lower():

        case "rr-nonlinear":
            ...

        case "rr-linear":
    
            class Model(
                LinearRadialReturnTangentialContact, Model
            ):
                """Add contact mechanics modeled as Alart linear radial return formulation."""

        case "ncp-min":
            
            class Model(
                ScaledContact,
                NCPNormalContact,  # Normal contact model
                NCPTangentialContact,  # Tangential contact model
                Model,
            ):
                """Contact mechanics modeled as NCP formulation."""
            model_params["ncp_type"] = "min"
            model_params["stick_slip_regularization"] = "origin_and_stick_slip_transition"

        case "ncp-fb":
            
            class Model(
                ScaledContact,
                NCPNormalContact,  # Normal contact model
                NCPTangentialContact,  # Tangential contact model
                Model,
            ):
                """Contact mechanics modeled as NCP formulation."""
            model_params["ncp_type"] = "fb"
            model_params["stick_slip_regularization"] = "origin_and_stick_slip_transition"

        case _:
            raise ValueError(f"Unknown formulation: {formulation}")
        

    # Adapt linearization method
    match linearization.lower():

        case "picard":
            ...

        case "newton":

            class Model(DarcysLawAd, Model):
                """Enhance with AD of permeability."""

        case _:
            raise ValueError(f"Unknown linearization method: {linearization}")

    # Add relaxation
    match relaxation.lower():
        case "none":
            ...

        case "linesearch":

            if formulation.lower() in ["rr-nonlinear", "rr-linear"]:
                class Model(
                    pp.models.solution_strategy.ContactIndicators,
                    Model,
                ):
                    """Added contact indicators for line search."""

            elif formulation.lower() in ["ncp-min", "ncp-fb-partial", "ncp-fb"]:
                class Model(
                    NCPContactIndicators,
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
            raise ValueError(f"Relaxation method {relaxation} not recognized.")

    # Choose linear solver
    match linear_solver.lower():
        case "scipy_sparse":
            # Use scipy sparse solver
            model_params["linear_solver"] = "scipy_sparse"
            solver_params["linear_solver"] = model_params["linear_solver"]

        case "pypardiso":
            # Use pypardiso solver
            model_params["linear_solver"] = "pypardiso"
            solver_params["linear_solver"] = model_params["linear_solver"]

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
            raise ValueError(f"Linear solver {linear_solver} not recognized.")

    return Model, model_params, solver_params