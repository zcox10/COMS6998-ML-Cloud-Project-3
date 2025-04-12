from fastapi import FastAPI
import logging

from src.dynamics.mpc import MPC
from src.dynamics.arm_dynamics_student import ArmDynamicsStudent
from src.model import MLP
from src.utils.utils import Utils
from src.robot.score import score_mpc_learnt_dynamics

app = FastAPI()

# Initialize global objects
controller_params = {
    "control_horizon": 1,
    "prediction_horizon": 1,
    "num_candidates": 50,
    "num_iterations": 10,
    "velocity_weight": 0.02,
    "distance_weight": 2.5,
    "data_collection": False,
}
dynamics_params = {
    "time_limit": 2.5,
    "num_links": 2,
    "link_mass": 0.1,
    "link_length": 1,
    "joint_viscous_friction": 0.1,
    "dt": 0.01,
    "dist_limit": [0.2, 0.3],
}
gcs_file_paths = {
    "bucket_name": "ml-cloud-kubeflow-pipeline-data",
    "model_path": "models",
    "eval_plot_path": "eval/plots",
    "eval_data_path": "eval/data",
    "training_data_path": "training_data",
}

utils = Utils()
controller = MPC(**controller_params)
dynamics = ArmDynamicsStudent(
    model_class=MLP(),
    gcs_file_paths=gcs_file_paths,
    **{
        k: dynamics_params[k]
        for k in ["num_links", "link_mass", "link_length", "joint_viscous_friction", "dt"]
    }
)
model = utils.load_gcs_file(
    gcs_file_paths["bucket_name"], gcs_file_paths["model_path"], file_suffix=".pth"
)

utils.configure_component_logging(log_level=logging.INFO)


@app.get("/")
def health_check():
    return {"message": "ML Cloud API is up and running!"}


@app.get("/predict")
def predict():
    logging.info("Predict endpoint called.")
    score_mpc_learnt_dynamics(controller=controller, arm_student=dynamics, model=model, gui=False)
    return {"status": "Prediction completed"}
