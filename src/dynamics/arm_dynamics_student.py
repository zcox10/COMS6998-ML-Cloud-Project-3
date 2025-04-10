import numpy as np
import torch

from ..robot.arm_dynamics_base import ArmDynamicsBase
from utils.utils import Utils


class ArmDynamicsStudent(ArmDynamicsBase):
    def __init__(
        self,
        device,
        model,
        data_path,
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
        self.device = device
        self.model = model

        # Load utils and retrieve mean/std of features and labels
        self.utils = Utils()
        self.utils.set_seed()
        self.X_mean, self.X_std, self.Y_mean, self.Y_std = self.utils.retrieve_dataset_statistics(
            data_path
        )

    def init_model(self, model_path, num_links=2, time_step=0.01, device=torch.device("cpu")):
        # Load model
        self.model.load_state_dict(torch.load(model_path))
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
        # print(f"\nNew Action: {new_action}; Shape: {new_action.shape}\n")
        return new_action
