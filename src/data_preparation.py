from kfp import dsl
from typing import Dict
import numpy as np
from multiprocessing import Pool, cpu_count

from src.robot.robot import Robot
from src.robot.arm_dynamics_teacher import ArmDynamicsTeacher
from src.dynamics.mpc import MPC


def _collect_data_chunk(args):
    """
    Collects training data for a 2-link robotic arm simulation over a chunk of goals.

    Each goal is used to simulate a trajectory of the robotic arm under Model Predictive Control (MPC).
    At each timestep, the arm state and control action are recorded. These form the training inputs (X)
    and outputs (Y) for a learned dynamics model. Success metrics (pass/fail) are also recorded based
    on how close the arm's end-effector is to the goal at the end of the trajectory.

    Args:
        args (tuple): A tuple containing:
            - goals (list[np.ndarray]): List of 2D goal positions for the end-effector.
            - controller_params (dict): Hyperparameters for the MPC controller (e.g. num_candidates, horizon).
            - dynamics_params (dict): Dynamics config including num_links, dt, joint friction, etc.

    Returns:
        tuple: (X_main, Y_main, passed_trials, partial_passed_trials, failed_trials)
            - X_main: np.ndarray of state-action pairs.
            - Y_main: np.ndarray of resulting next states.
            - passed_trials: Number of goals reached precisely.
            - partial_passed_trials: Goals nearly reached within a looser threshold.
            - failed_trials: Goals that were not sufficiently reached.
    """
    goals, controller_params, dynamics_params = args

    # Initialize arm simulator, dynamics simulator, and MPC controller
    arm = Robot(ArmDynamicsTeacher(**dynamics_params))
    dynamics = ArmDynamicsTeacher(**dynamics_params)
    controller = MPC(**controller_params)

    # Compute number of steps per trajectory and total training samples expected
    num_steps = round(dynamics_params["time_limit"] / dynamics_params["dt"])
    total_samples = (
        len(goals)
        * num_steps
        * controller_params["num_iterations"]
        * controller_params["num_candidates"]
    )

    # Allocate memory for state-action (X) and next state (Y) training samples
    X_main = np.zeros((total_samples, 6), dtype=np.float32)
    Y_main = np.zeros((total_samples, 4), dtype=np.float32)

    # Initialize performance tracking counters
    passed_trials = 0
    partial_passed_trials = 0
    failed_trials = 0

    # Iterate over each goal
    for i, goal in enumerate(goals):
        # Set initial state (arm straight down)
        initial_state = np.zeros((arm.dynamics.get_state_dim(), 1))
        initial_state[0] = -np.pi / 2.0
        arm.reset()
        arm.goal = goal
        arm.set_state(initial_state)
        action = np.zeros((arm.dynamics.get_action_dim(), 1))

        # Run simulation for each timestep
        for j in range(num_steps):
            old_state = arm.get_state()

            # Recompute control action according to control horizon
            if j % controller.control_horizon == 0:
                action, X_chunk, Y_chunk = controller.compute_action(
                    dynamics, old_state, goal, action
                )
                arm.set_action(action)

            # Advance simulation by one step
            arm.advance()
            new_state = arm.get_state()

            # Compute start/end indices to slice into X_main/Y_main
            start = (i * num_steps) + (
                controller_params["num_candidates"] * controller_params["num_iterations"] * j
            )
            end = start + (
                controller_params["num_candidates"] * controller_params["num_iterations"]
            )

            # Store computed training data
            X_main[start:end] = X_chunk
            Y_main[start:end] = Y_chunk

            # Check distance and velocity for success classification
            pos_ee = arm.dynamics.compute_fk(new_state)
            dist = np.linalg.norm(pos_ee - goal)
            vel_ee = np.linalg.norm(arm.dynamics.compute_vel_ee(old_state))

        # Track success/failure of the trial
        if dist < dynamics_params["dist_limit"][0] and vel_ee < 0.5:
            passed_trials += 1
        elif dist < dynamics_params["dist_limit"][1] and vel_ee < 0.5:
            partial_passed_trials += 1
        else:
            failed_trials += 1

    return X_main, Y_main, passed_trials, partial_passed_trials, failed_trials


@dsl.component(base_image="python:3.9-slim")
def data_preparation_component(
    num_tests: int, controller_params: dict, dynamics_params: dict, output_path: str
) -> Dict[str, str]:
    """
    A Kubeflow Pipelines DSL component for generating training data for a robotic arm simulation.

    This component:
    - Samples `num_tests` goal positions.
    - Parallelizes data collection using multiprocessing.
    - Saves the resulting input/output arrays (`X`, `Y`) to a specified output path as a pickle file.

    Args:
        num_tests (int): Number of goals to sample for simulation.
        controller_params (dict): Parameters for the MPC controller.
        dynamics_params (dict): Parameters for the arm dynamics.
        output_path (str): Destination path for the pickled data.

    Returns:
        Dict[str, str]: Dictionary containing the output path to the saved data under the key "data_path".
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import time
    import pickle
    import numpy as np
    from multiprocessing import Pool, cpu_count
    from pathlib import Path
    from robot.score import sample_goal
    import logging

    logging.basicConfig(level=logging.INFO)

    # Sample random goal positions
    goals = [sample_goal() for _ in range(num_tests)]

    # Determine number of parallel processes and chunk size
    num_procs = max(1, cpu_count() - 1)
    chunk_size = max(1, int(num_tests / (2 * num_procs)))

    # Split goals into chunks for each process
    goal_chunks = [goals[i : i + chunk_size] for i in range(0, len(goals), chunk_size)]
    args = [(chunk, controller_params, dynamics_params) for chunk in goal_chunks]

    # Collect training data in parallel
    start = time.time()
    with Pool(processes=num_procs) as pool:
        results = pool.map(_collect_data_chunk, args)

    # Combine results from all worker processes
    X = np.vstack([res[0] for res in results])  # Input features (state + action)
    Y = np.vstack([res[1] for res in results])  # Output targets (next state)
    data = {"X": X, "Y": Y}

    # Ensure output directory exists and save to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    # Upload to GCS as parquet file
    data = {
        **{f"x{i}": X[:, i] for i in range(X.shape[1])},
        **{f"y{i}": Y[:, i] for i in range(Y.shape[1])},
    }
    table = pa.table(data)
    pq.write_table(table, output_path)

    # Calculate metrics for logging
    passed_trials = sum([res[2] for res in results])
    partial_passed_trials = sum([res[3] for res in results])
    failed_trials = sum([res[4] for res in results])
    total_trials = passed_trials + partial_passed_trials + failed_trials

    # print metrics
    logging.info(
        f"\n{passed_trials} of {total_trials} trials passed ({round((passed_trials / total_trials) * 100, 2)}%)"
    )
    logging.info(
        f"{partial_passed_trials} of {total_trials} trials partially passed ({round((partial_passed_trials / total_trials) * 100, 2)}%)"
    )
    logging.info(
        f"{failed_trials} of {total_trials} trials failed ({round((failed_trials / total_trials) * 100, 2)}%)"
    )

    # Log runtime info
    logging.info("Saved data to:", output_path)
    logging.info(f"Finished in {(time.time() - start)/60:.2f} minutes")

    return {"data_path": output_path}
