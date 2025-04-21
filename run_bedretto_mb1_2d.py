"""Basic run script for 3d poromechanics simulation."""

import argparse
import logging
import time
from pathlib import Path

import porepy as pp
from bedretto_mb1_2d.physics import (
    BedrettoMB1_Model,
    solid_parameters,
    fluid_parameters,
    numerics_parameters,
    injection_schedule,
    BedrettoMB1_Model_Initialization,
    solid_parameters_initialization,
    fluid_parameters_initialization,
    numerics_parameters_initialization,
    injection_schedule_initialization,
)

from ncp import (
    AdvancedSolverStatistics,
)
from egc import setup_model
from copy import deepcopy

# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_case_name(
    num_fractures,
    formulation,
    linearization,
    relaxation,
    linear_solver,
    initialization=False,
):
    folder = Path(f"bedretto_mb1_{num_fractures}")
    name = f"{formulation.lower()}_{linearization.lower()}"
    if relaxation.lower() != "none":
        name += f"_{relaxation.lower()}"
    name += f"_{linear_solver.lower()}"
    path = folder / name
    if initialization:
        path = path / "initialization"
    return path


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
            schedule=injection_schedule["time"],
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

    (Model, model_params, solver_params) = setup_model(
        BedrettoMB1_Model,
        model_params,
        solver_params,
        formulation=args.formulation,
        linearization=args.linearization,
        relaxation=args.relaxation,
        linear_solver=args.linear_solver,
    )

    # Initialization step
    model_params_init = deepcopy(model_params)
    model_params_init["material_constants"] = {
        "solid": pp.SolidConstants(**solid_parameters_initialization),
        "fluid": pp.FluidComponent(**fluid_parameters_initialization),
        "numerical": pp.NumericalConstants(**numerics_parameters_initialization),
    }
    model_params_init["time_manager"] = pp.TimeManager(
        schedule=injection_schedule_initialization["time"],
        dt_init=pp.DAY,
        constant_dt=True,
    )
    model_params_init["folder_name"] = Path("visualization") / generate_case_name(
        args.num_fractures,
        args.formulation,
        args.linearization,
        args.relaxation,
        args.linear_solver,
        initialization=True,
    )

    (Model_Init, model_params_init, solver_params) = setup_model(
        BedrettoMB1_Model_Initialization,
        model_params_init,
        solver_params,
        formulation=args.formulation,
        linearization=args.linearization,
        relaxation=args.relaxation,
        linear_solver=args.linear_solver,
    )

    model_init = Model_Init(model_params_init)
    pp.run_time_dependent_model(
        model_init,
        solver_params,
    )

    # (Pre-)Initialize the model to build a bijection between domains
    model = Model(model_params)
    model.prepare_simulation_1()
    solver_params["prepare_simulation_1"] = False

    # TODO provide standardized routine to build bijection between domains

    # Build 1-1 map between domains
    unique_domains_init = {
        (domain.id, domain.dim, type(domain)): domain
        for domain in model_init.mdg.subdomains()
    }
    unique_interfaces_init = {
        (interface.id, interface.dim, type(interface)): interface
        for interface in model_init.mdg.interfaces()
    }
    unique_domains_init = dict(
        sorted(unique_domains_init.items(), key=lambda x: (x[0][0], x[0][1]))
    )
    unique_interfaces_init = dict(
        sorted(unique_interfaces_init.items(), key=lambda x: (x[0][0], x[0][1]))
    )

    # Same for current model
    unique_domains = {
        (domain.id, domain.dim, type(domain)): domain
        for domain in model.mdg.subdomains()
    }
    unique_interfaces = {
        (interface.id, interface.dim, type(interface)): interface
        for interface in model.mdg.interfaces()
    }
    unique_domains = dict(
        sorted(unique_domains.items(), key=lambda x: (x[0][0], x[0][1]))
    )
    unique_interfaces = dict(
        sorted(unique_interfaces.items(), key=lambda x: (x[0][0], x[0][1]))
    )

    # Build the domain bijection
    domain_bijection = dict(
        zip(
            list(unique_domains_init.values()) + list(unique_interfaces_init.values()),
            list(unique_domains.values()) + list(unique_interfaces.values()),
        )
    )

    # Set the initial condition for the model
    # model.params["initial_condition"] = model_init.equation_system.get_variable_values(iterate_index=0)
    model.params["initial_condition"] = {
        (
            var.name,
            domain_bijection[var.domain],
        ): model_init.equation_system.get_variable_values([var], iterate_index=0)
        for var in model_init.equation_system.variables
        # if var.name not in ["u", "u_interface"]
    }

    pp.run_time_dependent_model(model, solver_params)

    # Simple statistics
    logger.info(
        f"\nTotal number of iterations: {model_init.nonlinear_solver_statistics.cache_num_iteration}"
    )
    logger.info(
        f"\nTotal number of iterations: {model.nonlinear_solver_statistics.cache_num_iteration}"
    )
    logger.info(f"\nTotal time: {time.time() - t_0:.2f} s")
