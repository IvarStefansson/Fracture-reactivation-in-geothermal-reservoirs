import sys
import shutil
import numpy as np
import argparse
import subprocess
from pathlib import Path
import time

parser = argparse.ArgumentParser(description="Run single fracture test cases.")
parser.add_argument("-only-new", "--only-new", action="store_true", help="Only new.")
parser.add_argument("-dry", "--dry", action="store_true", help="Only dry run.")
parser.add_argument(
    "-c", "--cache", type=str, default="cache", help="cache for parallel runs."
)
parser.add_argument(
    "-p", "--parallel", action="store_true", help="run in parallel mode."
)
parser.add_argument(
    "-np",
    "--parallel-processors",
    type=int,
    nargs="+",
    default=[-1, -1],
    help="start and end.",
)
parser.add_argument(
    "-p-list",
    "--parallel-processors-list",
    type=int,
    nargs="+",
    default=[-1, -1],
    help="selected processor list",
)

args = parser.parse_args()

# ! ---- Fixed parameters ----

fractures = [
#    "7 8 9 10 11",
#    "7 8 9 10 11 12 13",
    "7 8 9 10 11 12 13 14",
]

injection = ["8w"] #, "8w",]

refinements = [2, 3] #,2, 3, 4]

dilation_angles = [0.05]

methods = [
    ("rr-nonlinear", "linesearch"),
    #("ncp-min", "linesearch"),
]

safety_measures = [""]

linear_solvers = ["pypardiso", "fthm"]

disks = ["large", "intermediate", "small"]

simple = [(False, False)]

linearization = "picard"
decoupling = False
iterative_decoupling = False
output = "visualization-paper-2-refined-updated-geometry-lower-dim-perm"

# ! ---- Options ----

if args.parallel:
    pool_instructions = []

for intervals in fractures:
    for disk in disks:
        for injection_interval in injection:
            for mesh_refinement in refinements:
                for dilation in dilation_angles:
                    for formulation, relaxation in methods:
                        for safety_measure in safety_measures:
                            for linear_solver in linear_solvers:
                                for simple_flow, tpfa_flow in simple:
                                    instructions = [
                                        sys.executable,
                                        "run_bedretto_valter.py",
                                        "--intervals",
                                        intervals,
                                        "--disks",
                                        disk,
                                        "--injection-interval",
                                        str(injection_interval),
                                        "--mesh-refinement",
                                        str(mesh_refinement),
                                        "--dilation",
                                        str(dilation),
                                        "--formulation",
                                        formulation,
                                        "--relaxation",
                                        relaxation,
                                        safety_measure,
                                        "--linear-solver",
                                        linear_solver,
                                        "--simple_flow" if simple_flow else "",
                                        "--tpfa_flow" if tpfa_flow else "",
                                        "--output",
                                        output,
                                    ]
                                    if args.only_new:
                                        # Check if the output file exists
                                        from run_bedretto_valter import generate_case_name

                                        safe_nrm = safety_measure == "--safe-nrm"
                                        safe_aa = safety_measure == "--safe-aa"
                                        safe_relaxation = safety_measure == "--safe-relaxation"

                                        output_file = (
                                            Path(output)
                                            / generate_case_name(
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
                                            )
                                            / "solver_statistics.json"
                                        )
                                        if output_file.exists():
                                            continue
                                        
                                    if args.parallel:
                                        pool_instructions.append(instructions)
                                    else:
                                        subprocess.run(instructions)

# Coordinate parallel runs using 'nohup taskset --cpu-list N python instructions (unrolled)'
# Use for N in range(args.parallel_processors[0], args.parallel_processors[1]+1)
# fill each cpu with a task, then move to next cpu after finished
if args.parallel:

    for i, pi in enumerate(pool_instructions):
        print(i, pi)

    # Distribute the tasks among the available processors
    available_cpus = [
        i for i in range(args.parallel_processors[0], args.parallel_processors[1] + 1)
    ]
    available_cpus += args.parallel_processors_list
    available_cpus = list(set(available_cpus))
    available_cpus = [proc for proc in available_cpus if proc > 0]
    num_available_cpus = len(available_cpus)
    split_instructions = {}
    for i, instruction in enumerate(pool_instructions):
        # Determine which cpu to use
        cpu = available_cpus[i % num_available_cpus]
        # Store the instruction in the split_instructions
        if cpu not in split_instructions:
            split_instructions[cpu] = []
        split_instructions[cpu].append(instruction)

    # Stop if dry run
    assert not (args.dry), "Only dry run."

    # Remove cache directory if it exists
    cache_dir = Path(args.cache)
    if cache_dir.exists():
        # Remove cache_dir using shutil
        shutil.rmtree(str(cache_dir))
    # Create cache directory if it does not exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Store the instructions to a file cache/parallel_instruction_processor_N.txt
    for cpu, instructions in split_instructions.items():
        with open(f"{args.cache}/parallel_instruction_processor_{cpu}.txt", "w") as f:
            for instruction in instructions:
                f.write(" ".join(instruction) + "\n")

    pool_instructions = list(cache_dir.glob("*.txt"))
    for i, instruction_file in enumerate(pool_instructions):
        # To mitigate racing conditions, wait 5 seconds for each additional run
        time.sleep(10)

        processor = instruction_file.stem.split("_")[-1]

        # Use nohup taskset to run the instruction file on the specified processor
        subprocess.Popen(
            " ".join(
                [
                    "nohup",
                    "taskset",
                    "--cpu-list",
                    str(int(processor) - 1),
                    sys.executable,
                    "run_instructions.py",
                    "--path",
                    str(instruction_file),
                    ">",
                    f"nohup_{processor}.out",
                    "2>&1",
                    "&",
                ]
            ),
            shell=True,
        )
