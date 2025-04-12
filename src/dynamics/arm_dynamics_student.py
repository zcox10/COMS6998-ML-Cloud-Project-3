import numpy as np
import torch
import logging
from src.robot.arm_dynamics_base import ArmDynamicsBase
from src.utils.utils import Utils


class ArmDynamicsStudent(ArmDynamicsBase):
    def __init__(
        self,
        model_class,
        gcs_file_paths,
        num_links,
        link_mass,
        link_length,
        joint_viscous_friction,
        dt,
    ):
        super().__init__(
            num_links=num_links,
            link_mass=link_mass,
            link_length=link_length,
            joint_viscous_friction=joint_viscous_friction,
            dt=dt,
        )
        self.utils = Utils()
        self.device = self.utils.set_device()
        self.model = model_class

        # Load utils and retrieve mean/std of features and labels
        data = self.utils.load_gcs_file(
            gcs_file_paths["bucket_name"],
            gcs_file_paths["training_data_path"],
            file_suffix=".parquet",
        )
        self.X_mean, self.X_std, self.Y_mean, self.Y_std = self.utils.retrieve_dataset_statistics(
            data
        )

    def init_model(self, model):
        # Load model
        self.model.load_state_dict(model)
        self.model.eval()
        self.model_loaded = True

    def dynamics_step(self, state, action, dt):
        if not self.model_loaded:
            raise ValueError("You must load the model via init_model().")

        # Transform state and action into X feature
        X = np.concatenate([state.reshape(1, -1), action.reshape(1, -1)], axis=1)
        X_norm = self.utils.normalize_input(X, self.X_mean, self.X_std)
        X_tensor = torch.from_numpy(X_norm).float().to(self.device)

        # Generate prediction
        with torch.no_grad():
            y_pred_norm = self.model(X_tensor).cpu().numpy()

        # De-normalize prediction according to X, Y_mean, and Y_std
        new_action = self.utils.denormalize_prediction(
            X, y_pred_norm, self.Y_mean, self.Y_std
        ).reshape(4, 1)

        # logging.debug(f"\nNew Action: {new_action}; Shape: {new_action.shape}\n")
        return new_action
