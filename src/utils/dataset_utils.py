import numpy as np
import torch
import logging
from torch.utils.data import DataLoader, Dataset, random_split


class DynamicDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class DatasetUtils:
    def __init__(self):
        # X.shape (N, 6); Y.shape (N, 4)
        self.X = None
        self.Y = None

        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None

    def _calculate_data_statistics(self, data):
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        return mean, std

    def _normalize_data(self, data, mean, std):
        """
        Normalize data and return norm_data, mean, and std.
        """
        norm_data = (data - mean) / std
        return norm_data

    def _generate_features(self):
        # Normalize X and Y and store their respective mean and std
        self.X_mean, self.X_std = self._calculate_data_statistics(self.X)
        X_norm = self._normalize_data(self.X, self.X_mean, self.X_std)

        self.Y_mean, self.Y_std = self._calculate_data_statistics(self.Y)
        Y_norm = self._normalize_data(self.Y, self.Y_mean, self.Y_std)

        # Print statistics
        logging.debug(f"X_mean: {self.X_mean}")
        logging.debug(f"X_std: {self.X_std}")
        logging.debug(f"Y_mean: {self.Y_mean}")
        logging.debug(f"Y_std: {self.Y_std}")

        return X_norm, Y_norm

    def _generate_data(self, data):
        self.X = data["X"].astype(np.float32)
        self.Y = data["Y"].astype(np.float32)

    def retrieve_dataset_statistics(self, data):
        self._generate_data(data)
        _, _ = self._generate_features()
        return self.X_mean, self.X_std, self.Y_mean, self.Y_std

    def normalize_input(self, X_input, X_mean, X_std):
        """
        Normalize a single input sample (shape: [1, 6]) using stored training stats.
        """
        return (X_input - X_mean) / X_std

    def denormalize_prediction(self, X_input, Y_pred_norm, Y_mean, Y_std):
        """
        Convert normalized Y prediction back to real-world next state.
        """
        # Output next state
        Y = Y_pred_norm * Y_std + Y_mean
        return Y

    def create_data_loaders(self, data, batch_size, train_ratio, val_ratio, seed):
        self._generate_data(data)

        # Normalize data and generate Y
        X_norm, Y_norm = self._generate_features()
        dataset = DynamicDataset(X_norm, Y_norm)

        # Shuffle data (and optionally, select top [:1000])
        generator = torch.Generator().manual_seed(seed)
        # all_indices = torch.randperm(len(dataset), generator=generator)[:1000]
        all_indices = torch.randperm(len(dataset), generator=generator)
        dataset = torch.utils.data.Subset(dataset, all_indices)

        # Calculate train, val, test sizes
        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size

        # Create random splits
        train_set, val_set, test_set = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

        # Create dataloaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
