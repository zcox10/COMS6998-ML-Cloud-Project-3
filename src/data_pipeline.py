import os
import time
from datetime import datetime
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count
from google.cloud import storage
import logging
from typing import Dict

from src.robot.score import sample_goal
from src.robot.robot import Robot
from src.robot.arm_dynamics_teacher import ArmDynamicsTeacher
from src.dynamics.mpc import MPC
from src.utils.utils import Utils


class DataPipeline:
    def __init__(self):
        self.utils = Utils()

    @staticmethod
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
        arm = Robot(
            ArmDynamicsTeacher(
                num_links=dynamics_params["num_links"],
                link_mass=dynamics_params["link_mass"],
                link_length=dynamics_params["link_length"],
                joint_viscous_friction=dynamics_params["joint_viscous_friction"],
                dt=dynamics_params["dt"],
            )
        )
        dynamics = ArmDynamicsTeacher(
            num_links=dynamics_params["num_links"],
            link_mass=dynamics_params["link_mass"],
            link_length=dynamics_params["link_length"],
            joint_viscous_friction=dynamics_params["joint_viscous_friction"],
            dt=dynamics_params["dt"],
        )
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

    def run_data_pipeline(
        self,
        num_tests: int,
        controller_params: dict,
        dynamics_params: dict,
        gcs_file_paths: Dict[str, str],
    ) -> Dict[str, str]:
        """
        A data processing component for generating training data for a robotic arm simulation.

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
        logging.debug("Data prepartion imports are complete")

        # Sample random goal positions
        goals = [sample_goal() for _ in range(num_tests)]

        # Determine number of parallel processes and chunk size
        num_procs = max(1, cpu_count() - 1)
        chunk_size = max(1, int(num_tests / (2 * num_procs)))

        # Split goals into chunks for each process
        goal_chunks = [goals[i : i + chunk_size] for i in range(0, len(goals), chunk_size)]
        args = [(chunk, controller_params, dynamics_params) for chunk in goal_chunks]

        # Collect training data in parallel
        logging.debug("Going inside _collect_data_chunk")
        start = time.time()
        with Pool(processes=num_procs) as pool:
            results = pool.map(DataPipeline._collect_data_chunk, args)

        # Combine results from all worker processes
        X = np.vstack([res[0] for res in results])  # Input features (state + action)
        Y = np.vstack([res[1] for res in results])  # Output targets (next state)
        data = {
            "X": [row.tolist() for row in X],
            "Y": [row.tolist() for row in Y],
        }

        # if isinstance(table, pa.Table):

        # Save as local copy, upload to GCS as parquet file, then delete local copy
        table = pa.table(data)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename_prefix = f"training_data_{timestamp}"

        gcs_uri = self.utils.save_asset_to_gcs(
            table,
            gcs_file_paths["bucket_name"],
            gcs_file_paths["training_data_path"],
            filename_prefix,
        )

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
        logging.info("Saved data to:", gcs_uri)
        logging.info(f"Finished in {(time.time() - start)/60:.2f} minutes")

        return {"data_path": gcs_uri}
