import subprocess

formulations = [
    "rr-nonlinear",
    "rr-linear",
    "ncp-min",
    "ncp-fb"
]

linear_solvers = [
    "scipy_sparse",
    "pypardiso",
    "fthm"
]

relaxations = [
    "none",
    "linesearch",
    "return-map",
]

def run_solver(formulation, linear_solver, relaxation):
    command = [
        "python3", "run_simple_bedretto.py",
        "--formulation", formulation,
        "--linear-solver", linear_solver,
        "--relaxation", relaxation
    ]
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

def main():
    for formulation in formulations:
        for linear_solver in linear_solvers:
            for relaxation in relaxations:
                run_solver(formulation, linear_solver, relaxation)

if __name__ == "__main__":
    main()