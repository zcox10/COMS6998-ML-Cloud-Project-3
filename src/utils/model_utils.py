import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import logging
from src.utils.generic_utils import GenericUtils


class ModelUtils:
    def __init__(self):
        pass

    def train_regression_loop(
        self,
        epochs,
        model,
        criterion,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        patience,
        clip_value,
        device,
        monitor_gradients,
        early_stop,
    ):
        """
        Training loop for an NN. Returns metrics for plotting and regression reports.

        Returns:
        - training_metrics
        - validation_metrics
        - learning_rates
        """

        # Store all training predictions and labels
        train_losses, val_losses = [], []  # losses according to criterion provided
        train_mses, val_mses = [], []
        train_rmses, val_rmses = [], []
        train_maes, val_maes = [], []
        train_r2s, val_r2s = [], []
        learning_rates = []

        best_val_loss = float("inf")
        patience_counter = 0  # early stopping

        # Move model to specified device
        model.to(device)

        epoch_predictions = []
        epoch_labels = []
        for epoch in range(epochs):
            model.train()
            total_loss = torch.zeros(1, dtype=torch.float32, device=device)
            mse_sum = torch.zeros(1, dtype=torch.float32, device=device)  # sum of squared errors
            mae_sum = torch.zeros(1, dtype=torch.float32, device=device)  # sum of absolute errors
            s_y = torch.zeros(1, dtype=torch.float32, device=device)  # sum of labels
            s_y2 = torch.zeros(1, dtype=torch.float32, device=device)  # sum of labels^2
            total_samples = 0

            # Training loop
            train_metrics = {}
            for inputs, labels in train_dataloader:
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device).float()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # gradient clipping
                optimizer.step()
                batch_numel = labels.numel()
                total_loss += loss * batch_numel
                total_samples += batch_numel

                diff = outputs - labels
                mse_sum += (diff**2).sum()
                mae_sum += diff.abs().sum()
                s_y += labels.sum()
                s_y2 += (labels**2).sum()

            # Compute and add training metrics
            avg_loss = total_loss.item() / total_samples
            mse = mse_sum.item() / total_samples
            rmse = np.sqrt(mse)
            mae = mae_sum.item() / total_samples

            # Calculate R2
            tss = s_y2.item() - ((s_y.item() ** 2) / total_samples)
            r2 = 1.0 - (mse_sum.item() / tss) if tss != 0 else 0.0

            train_losses.append(avg_loss)
            train_mses.append(mse)
            train_rmses.append(rmse)
            train_maes.append(mae)
            train_r2s.append(r2)

            # Run validation and store results
            val_metrics = self._validation_regression_loop(model, criterion, val_dataloader, device)

            # Append scalar values
            val_losses.append(val_metrics["avg_loss"])
            val_mses.append(val_metrics["mse"])
            val_rmses.append(val_metrics["rmse"])
            val_maes.append(val_metrics["mae"])
            val_r2s.append(val_metrics["r2"])

            # Early stopping logic
            if early_stop:
                patience_counter, best_val_loss = self._determine_early_stopping(
                    val_metrics["avg_loss"], best_val_loss, patience_counter
                )
                # logging.info(f"Patience counter: {patience_counter}")
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered! Epoch {epoch + 1}")
                    break

            # Print epoch summary
            if (epoch + 1) % 5 == 0:
                if monitor_gradients:
                    avg_grad = self._monitor_gradients(model)

                # For spacing formatting in print statement
                train_fmt = "<7.6f"
                val_fmt = "10.6f"

                # Print training metrics every N epochs
                progress_message = (
                    f"\n================================================ "
                    + f"Epoch {epoch+1:>2}/{epochs:<2}"
                    + " ================================================\n"
                    + f"Train Loss: {avg_loss:{train_fmt}} | "
                    + f"Train MSE: {mse:{train_fmt}} | "
                    + f"Train RMSE: {rmse:{train_fmt}} | "
                    + f"Train MAE: {mae:{train_fmt}} | "
                    + f"Train R²: {r2:{train_fmt}}\n"
                    + f"Val Loss: {val_metrics['avg_loss']:{val_fmt}} | "
                    + f"Val MSE: {val_metrics['mse']:{val_fmt}} | "
                    + f"Val RMSE: {val_metrics['rmse']:{val_fmt}} | "
                    + f"Val MAE: {val_metrics['mae']:{val_fmt}} | "
                    + f"Val R²: {val_metrics['r2']:{val_fmt}}"
                )
                progress_message += (
                    f"\nAvg Gradient: {avg_grad:<10.8f}" if monitor_gradients else ""
                )
                logging.info(progress_message)

            # Update learning rate according to scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["avg_loss"])
            elif scheduler is not None:
                scheduler.step()

            # Add learning rate
            for param_group in optimizer.param_groups:
                learning_rates.append(param_group["lr"])

        logging.info("Training complete")
        return {
            # Train Epoch Metrics
            "train_losses": train_losses,
            "train_mses": train_mses,
            "train_rmses": train_rmses,
            "train_maes": train_maes,
            "train_r2s": train_r2s,
            # Val Epoch Metrics
            "val_losses": val_losses,
            "val_mses": val_mses,
            "val_rmses": val_rmses,
            "val_maes": val_maes,
            "val_r2s": val_r2s,
            # Final Train Metrics
            "final_train_loss": np.mean(train_losses),
            "final_train_mse": np.mean(train_mses),
            "final_train_rmse": np.mean(train_rmses),
            "final_train_mae": np.mean(train_maes),
            "final_train_r2": np.mean(train_r2s),
            # Final Val Metrics
            "final_val_loss": np.mean(val_losses),
            "final_val_mse": np.mean(val_mses),
            "final_val_rmse": np.mean(val_rmses),
            "final_val_mae": np.mean(val_maes),
            "final_val_r2": np.mean(val_r2s),
            # Other
            "learning_rates": learning_rates,
            "best_val_loss": best_val_loss,
            "final_epoch": epoch + 1,
        }

    def _monitor_gradients(self, model):
        # Monitor for vanishing or exploding gradients
        total_gradients = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_gradients.append(grad_norm)
                if grad_norm > 1e8:
                    logging.info(
                        f"\nWarning: Exploding Gradient Detected in {name}: {grad_norm:.2e}\n"
                    )
                elif grad_norm < 1e-8:
                    logging.info(f"Warning: Vanishing Gradient Detected in {name}: {grad_norm}")
        avg_grad = sum(total_gradients) / len(total_gradients) if total_gradients else 0
        return avg_grad

    def _validation_regression_loop(self, model, criterion, val_dataloader, device):
        """
        Validation loop and storing predictions.
        """

        model.eval()

        total_loss = torch.zeros(1, dtype=torch.float32, device=device)
        mse_sum = torch.zeros(1, dtype=torch.float32, device=device)  # sum of squared errors
        mae_sum = torch.zeros(1, dtype=torch.float32, device=device)  # sum of absolute errors
        s_y = torch.zeros(1, dtype=torch.float32, device=device)  # sum of labels
        s_y2 = torch.zeros(1, dtype=torch.float32, device=device)  # sum of labels^2
        total_samples = 0

        # Validation loop
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device).float()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Calculate metrics
                batch_numel = labels.numel()
                total_loss += loss * batch_numel
                total_samples += batch_numel

                diff = outputs - labels
                mse_sum += (diff**2).sum()
                mae_sum += diff.abs().sum()
                s_y += labels.sum()
                s_y2 += (labels**2).sum()

        # Compute and add training metrics
        avg_loss = total_loss.item() / total_samples
        mse = mse_sum.item() / total_samples
        rmse = np.sqrt(mse)
        mae = mae_sum.item() / total_samples

        # Calculate R2
        tss = s_y2.item() - ((s_y.item() ** 2) / total_samples)
        r2 = 1.0 - (mse_sum.item() / tss) if tss != 0 else 0.0

        # Return metrics
        return {
            "avg_loss": avg_loss,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    def _determine_early_stopping(self, avg_val_loss, best_val_loss, patience_counter):
        """
        Checks if early stopping condition is met.
        """
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        return patience_counter, best_val_loss

    def _calculate_test_regression_performance(self, model, dataloader, criterion, device):
        """
        Evaluate a model on a test dataset to determine its performance.
        """
        model.to(device)
        model.eval()

        total_loss = torch.zeros(1, dtype=torch.float32, device=device)
        mse_sum = torch.zeros(1, dtype=torch.float32, device=device)  # sum of squared errors
        mae_sum = torch.zeros(1, dtype=torch.float32, device=device)  # sum of absolute errors
        s_y = torch.zeros(1, dtype=torch.float32, device=device)  # sum of labels
        s_y2 = torch.zeros(1, dtype=torch.float32, device=device)  # sum of labels^2
        total_samples = 0

        logging.info("Calculate Test Performance")
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device).float()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Calculate metrics
                batch_numel = labels.numel()
                total_loss += loss * batch_numel
                total_samples += batch_numel

                diff = outputs - labels
                mse_sum += (diff**2).sum()
                mae_sum += diff.abs().sum()
                s_y += labels.sum()
                s_y2 += (labels**2).sum()

        # Compute and add training metrics
        avg_loss = total_loss.item() / total_samples
        mse = mse_sum.item() / total_samples
        rmse = np.sqrt(mse)
        mae = mae_sum.item() / total_samples

        # Calculate R2
        tss = s_y2.item() - ((s_y.item() ** 2) / total_samples)
        r2 = 1.0 - (mse_sum.item() / tss) if tss != 0 else 0.0

        fmt = "<7.6f"
        test_metrics_message = (
            f"\n==================== Test Metrics ====================\n"
            + f"Train Loss: {avg_loss:{fmt}} | "
            + f"Train MSE: {mse:{fmt}} | "
            + f"Train RMSE: {rmse:{fmt}} | "
            + f"Train MAE: {mae:{fmt}} | "
            + f"Train R²: {r2:{fmt}}\n"
        )
        logging.info(test_metrics_message)

        return {
            "final_test_loss": avg_loss,
            "final_test_mse": mse,
            "final_test_rmse": rmse,
            "final_test_mae": mae,
            "final_test_r2": r2,
        }

    def _plot_training_regression_summary(
        self, training_metrics, test_metrics, gcs_file_paths, show_plot
    ):
        """
        Plots a 3x3 summary:
          - Row 1: Loss, MSE, RMSE (Train vs Validation)
          - Row 2: MAE, R2 (Train vs Validation), and Learning Rate
          - Row 3: Final metrics text for Train, Validation, and Test
        """

        # Extract arrays for plotting
        train_losses = training_metrics["train_losses"]
        val_losses = training_metrics["val_losses"]
        train_mses = training_metrics["train_mses"]
        val_mses = training_metrics["val_mses"]
        train_rmses = training_metrics["train_rmses"]
        val_rmses = training_metrics["val_rmses"]
        train_maes = training_metrics["train_maes"]
        val_maes = training_metrics["val_maes"]
        train_r2s = training_metrics["train_r2s"]
        val_r2s = training_metrics["val_r2s"]
        lrs = training_metrics["learning_rates"]

        # Prepare x-axis for the epoch-based metrics
        epochs = np.arange(1, len(train_losses) + 1)
        lr_epochs = np.arange(1, len(lrs) + 1)

        # Create figure and axes: 3 rows, 3 columns
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle("Training Summary", fontsize=16, y=0.97)

        # Row 1: Loss, MSE, RMSE
        # define colors for plots
        train_color = "blue"
        val_color = "red"
        lr_color = "darkgreen"

        # (0, 0): Loss
        axes[0, 0].plot(epochs, train_losses, marker="o", color=train_color, label="Train")
        axes[0, 0].plot(epochs, val_losses, marker="o", color=val_color, label="Val")
        axes[0, 0].set_title("Loss vs. Epoch")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)
        axes[0, 0].legend()

        # (0, 1): MSE
        axes[0, 1].plot(epochs, train_mses, marker="o", color=train_color, label="Train")
        axes[0, 1].plot(epochs, val_mses, marker="o", color=val_color, label="Val")
        axes[0, 1].set_title("MSE vs. Epoch")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MSE")
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        # (0, 2): RMSE
        axes[0, 2].plot(epochs, train_rmses, marker="o", color=train_color, label="Train")
        axes[0, 2].plot(epochs, val_rmses, marker="o", color=val_color, label="Val")
        axes[0, 2].set_title("RMSE vs. Epoch")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("RMSE")
        axes[0, 2].grid(True)
        axes[0, 2].legend()

        # Row 2: MAE, R2, Learning Rate
        # (1, 0): MAE
        axes[1, 0].plot(epochs, train_maes, marker="o", color=train_color, label="Train")
        axes[1, 0].plot(epochs, val_maes, marker="o", color=val_color, label="Val")
        axes[1, 0].set_title("MAE vs. Epoch")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("MAE")
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        # (1, 1): R2
        axes[1, 1].plot(epochs, train_r2s, marker="o", color=train_color, label="Train")
        axes[1, 1].plot(epochs, val_r2s, marker="o", color=val_color, label="Val")
        axes[1, 1].set_title("R² vs. Epoch")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("R² Score")
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        # (1, 2): Learning Rate
        axes[1, 2].plot(lr_epochs, lrs, marker="o", color=lr_color, label="Learning Rate")
        axes[1, 2].set_title("Learning Rate vs. Step")
        axes[1, 2].set_xlabel("Update Step")
        axes[1, 2].set_ylabel("Learning Rate")
        axes[1, 2].grid(True)
        axes[1, 2].legend()

        # Row 3: Final metrics text
        # Set formatting for each metric
        fmt = f"<7.6f"

        # Collect final training metrics
        text_col_1 = (
            f"Training Metrics:\n"
            f"\nAvg Loss:\n"
            f"- Train: {training_metrics['final_train_loss']:{fmt}}\n"
            f"- Val:   {training_metrics['final_val_loss']:{fmt}}\n"
            f"- Test:  {test_metrics['final_test_loss']:{fmt}}\n"
            f"\nMSE:\n"
            f"- Train: {training_metrics['final_train_mse']:{fmt}}\n"
            f"- Val:   {training_metrics['final_val_mse']:{fmt}}\n"
            f"- Test:  {test_metrics['final_test_mse']:{fmt}}\n"
        )
        text_col_2 = (
            f"Training Metrics:\n"
            f"\nRMSE:\n"
            f"- Train: {training_metrics['final_train_rmse']:{fmt}}\n"
            f"- Val:   {training_metrics['final_val_rmse']:{fmt}}\n"
            f"- Test:  {test_metrics['final_test_rmse']:{fmt}}\n"
            f"\nMAE:\n"
            f"- Train: {training_metrics['final_train_mae']:{fmt}}\n"
            f"- Val:   {training_metrics['final_val_mae']:{fmt}}\n"
            f"- Test:  {test_metrics['final_test_mae']:{fmt}}\n"
        )

        # Collect final validation metrics
        text_col_3 = (
            f"Training Metrics:\n"
            f"\nR²:\n"
            f"- Train: {training_metrics['final_train_r2']:{fmt}}\n"
            f"- Val:   {training_metrics['final_val_r2']:{fmt}}\n"
            f"- Test:  {test_metrics['final_test_r2']:{fmt}}\n"
            f"\nFinal Epoch: {training_metrics['final_epoch']}\n"
            f"\nBest Val Loss: {training_metrics['best_val_loss']:{fmt}}\n"
        )

        # Third row (2, 0)
        axes[2, 0].axis("off")
        axes[2, 0].text(
            0.02,
            0.98,
            text_col_1,
            va="top",
            ha="left",
            fontsize=11,
            fontfamily="monospace",
            wrap=True,
        )

        # Third row (2, 1)
        axes[2, 1].axis("off")
        axes[2, 1].text(
            0.02,
            0.98,
            text_col_2,
            va="top",
            ha="left",
            fontsize=11,
            fontfamily="monospace",
            wrap=True,
        )

        # Third row (2, 2)
        axes[2, 2].axis("off")
        axes[2, 2].text(
            0.02,
            0.98,
            text_col_3,
            va="top",
            ha="left",
            fontsize=11,
            fontfamily="monospace",
            wrap=True,
        )

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        gcs_uri = GenericUtils().save_asset_to_gcs(
            fig,
            gcs_file_paths["bucket_name"],
            gcs_file_paths["eval_plot_path"],
            gcs_file_paths["filename_prefix"],
        )

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def evaluate_regression_model(
        self,
        model,
        dataloader,
        criterion,
        device,
        training_metrics,
        total_samples,
        model_parameters,
        gcs_file_paths,
        show_plot,
    ):
        # Calculate test performance
        test_metrics = self._calculate_test_regression_performance(
            model, dataloader, criterion, device
        )

        # Generate plot summary
        self._plot_training_regression_summary(
            training_metrics=training_metrics,
            test_metrics=test_metrics,
            gcs_file_paths=gcs_file_paths,
            show_plot=show_plot,
        )

        # Save model parameters and performance to CSV
        df = pd.DataFrame([model_parameters])

        # Insert any fields you want into df
        df["file_name"] = [gcs_file_paths["filename_prefix"]]
        df["total_samples"] = [total_samples]
        df["final_epoch"] = [training_metrics["final_epoch"]]
        df["best_val_loss"] = [training_metrics["best_val_loss"]]
        df["train_loss"] = [training_metrics["final_train_loss"]]
        df["val_loss"] = [training_metrics["final_val_loss"]]
        df["test_loss"] = [test_metrics["final_test_loss"]]
        df["train_mse"] = [training_metrics["final_train_mse"]]
        df["val_mse"] = [training_metrics["final_val_mse"]]
        df["test_mse"] = [test_metrics["final_test_mse"]]
        df["train_rmse"] = [training_metrics["final_train_rmse"]]
        df["val_rmse"] = [training_metrics["final_val_rmse"]]
        df["test_rmse"] = [test_metrics["final_test_rmse"]]
        df["train_mae"] = [training_metrics["final_train_mae"]]
        df["val_mae"] = [training_metrics["final_val_mae"]]
        df["test_mae"] = [test_metrics["final_test_mae"]]
        df["train_r2"] = [training_metrics["final_train_r2"]]
        df["val_r2"] = [training_metrics["final_val_r2"]]
        df["test_r2"] = [test_metrics["final_test_r2"]]

        # Safe df
        gcs_uri = GenericUtils().save_asset_to_gcs(
            df,
            gcs_file_paths["bucket_name"],
            gcs_file_paths["eval_data_path"],
            gcs_file_paths["filename_prefix"],
        )

        return df
