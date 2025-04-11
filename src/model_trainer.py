import time
from datetime import datetime
import torch
import torch.nn as nn
from typing import Dict
import logging
from src.utils.utils import Utils


class MLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=4):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.block1 = nn.Linear(hidden_dim, hidden_dim)
        self.block2 = nn.Linear(hidden_dim, hidden_dim)
        self.block3 = nn.Linear(hidden_dim, hidden_dim)
        self.block4 = nn.Linear(hidden_dim, hidden_dim)
        self.block5 = nn.Linear(hidden_dim, hidden_dim)
        self.block6 = nn.Linear(hidden_dim, hidden_dim)
        self.block7 = nn.Linear(hidden_dim, hidden_dim)
        self.block8 = nn.Linear(hidden_dim, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input to hidden layer
        x = self.relu(self.input_layer(x))

        # residual block
        x1 = self.relu(self.block1(x))
        x2 = self.block2(x1)
        x = self.relu(x + x2)

        # residual block
        x3 = self.relu(self.block3(x))
        x4 = self.block4(x3)
        x = self.relu(x + x4)

        # residual block
        x5 = self.relu(self.block5(x))
        x6 = self.block6(x5)
        x = self.relu(x + x6)

        # residual block
        x7 = self.relu(self.block7(x))
        x8 = self.block8(x7)
        x = self.relu(x + x8)

        # output layer
        out = self.output_layer(x)
        return out


class ModelTrainer:
    def __init__(self, device):
        # Time training loop
        self.start = time.perf_counter()
        self.monitor_gradients = False
        self.early_stop = True

        # Import Utils()
        self.utils = Utils()
        self.utils.set_seed()

        # Model params
        self.batch_size = 512
        self.epochs = 60
        self.patience = 10
        self.clip_value = 1.0
        self.learning_rate = 0.003  # optimizer

        self.criterion = torch.nn.MSELoss()
        self.device = self.utils.set_device(device)

        # init model
        self.model = MLP()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.003
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.model_parameters = {
            "model_architecture": str(self.model),
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "criterion": str(self.criterion),
            "optimizer": str(self.optimizer),
            "scheduler": (
                f"{self.scheduler.__class__.__name__}({vars(self.scheduler)})"
                if self.scheduler
                else None
            ),
            "patience": self.patience,
            "clip_value": self.clip_value,
            "early_stop": self.early_stop,
        }

    def train(
        self,
        show_plot: bool,
        gcs_file_paths: Dict[str, str],
    ):

        data = self.utils.retrieve_latest_gcs_parquet_file(
            gcs_file_paths["bucket_name"], gcs_file_paths["training_data_path"]
        )

        logging.info("Start training loop")
        train_loader, val_loader, test_loader = self.utils.create_data_loaders(
            data=data, batch_size=self.batch_size, train_ratio=0.8, val_ratio=0.1, seed=10
        )
        total_samples = (
            len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        )
        logging.info(f"Total Samples: {total_samples}")
        logging.info(f"Train Loader Samples: {len(train_loader.dataset)}")
        logging.info(f"Val Loader Samples: {len(val_loader.dataset)}")
        logging.info(f"Test Loader Samples: {len(test_loader.dataset)}")

        # Training
        training_metrics = self.utils.train_regression_loop(
            epochs=self.epochs,
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            patience=self.patience,
            clip_value=self.clip_value,
            device=self.device,
            monitor_gradients=self.monitor_gradients,
            early_stop=self.early_stop,
        )

        # Evaluation
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        gcs_file_paths["filename_prefix"] = f"model_{current_time}"

        df = self.utils.evaluate_regression_model(
            model=self.model,
            dataloader=test_loader,
            criterion=self.criterion,
            device=self.device,
            training_metrics=training_metrics,
            total_samples=total_samples,
            model_parameters=self.model_parameters,
            gcs_file_paths=gcs_file_paths,
            show_plot=show_plot,
        )

        # Save model
        gcs_uri = self.utils.save_asset_to_gcs(
            self.model,
            gcs_file_paths["bucket_name"],
            gcs_file_paths["model_path"],
            gcs_file_paths["filename_prefix"],
        )

        # Output training time
        _ = self.utils.time_operation(self.start, message=f"Total training time")

        return {"model_path": gcs_uri}
