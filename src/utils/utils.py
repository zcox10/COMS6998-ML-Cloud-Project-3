from ..utils.dataset_utils import DatasetUtils
from ..utils.generic_utils import GenericUtils
from ..utils.model_utils import ModelUtils


class Utils:
    def __init__(self):
        self.generic_utils = GenericUtils()
        self.dataset_utils = DatasetUtils()
        self.model_utils = ModelUtils()

    # GenericUtils
    def configure_component_logging(self, log_level):
        self.generic_utils.configure_component_logging(log_level)

    def set_seed(self, seed=10):
        self.generic_utils.set_seed(seed)

    def set_device(self, device):
        self.generic_utils.set_device(device)

    def time_operation(self, start, message="Elapsed time"):
        return self.generic_utils.time_operation(start, message)

    def create_dirs(self, dirs_to_create):
        self.generic_utils.create_dirs(dirs_to_create)

    def add_scores_to_df(self, scores, directory, prefix, extension):
        return self.generic_utils.add_scores_to_df(scores, directory, prefix, extension)

    def get_highest_sorted_file(self, directory, prefix, extension):
        return self.generic_utils.get_highest_sorted_file(directory, prefix, extension)

    def view_metrics(self, data_directory, plots_directory, prefix, sort_cols):
        return self.generic_utils.view_metrics(data_directory, plots_directory, prefix, sort_cols)

    # DatasetUtils
    def retrieve_dataset_statistics(self, data_file):
        return self.dataset_utils.retrieve_dataset_statistics(data_file)

    def normalize_input(self, X_input, X_mean, X_std):
        return self.dataset_utils.normalize_input(X_input, X_mean, X_std)

    def denormalize_prediction(self, X_input, Y_pred_norm, Y_mean, Y_std):
        return self.dataset_utils.denormalize_prediction(X_input, Y_pred_norm, Y_mean, Y_std)

    def create_data_loaders(self, data_file, batch_size, train_ratio, val_ratio, seed):
        return self.dataset_utils.create_data_loaders(
            data_file, batch_size, train_ratio, val_ratio, seed
        )

    # ModelUtils
    def train_regression_loop(self, **kwargs):
        return self.model_utils.train_regression_loop(**kwargs)

    def evaluate_regression_model(self, **kwargs):
        return self.model_utils.evaluate_regression_model(**kwargs)
