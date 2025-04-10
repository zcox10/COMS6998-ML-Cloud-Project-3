import torch
import time
import os
import re
import glob
import ast
import numpy as np
import pandas as pd
from IPython.display import Image, display


class GenericUtils:
    def __init__(self):
        pass

    def set_seed(self, seed):
        """
        Sets a random seed throughout the notebook.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def set_device(self, device):
        return torch.device(device)

    def time_operation(self, start, message):
        end = time.perf_counter()
        elapsed = end - start
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"\n{message}: {minutes} min {seconds:.2f} sec\n")
        return round(elapsed / 60, 2)

    def get_highest_sorted_file(self, directory, prefix, extension):
        # Gather all CSV files starting with prefix and located in directory
        files = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and f.startswith(prefix)
            and f.endswith(extension)
        ]

        if not files:
            return None

        # Sort files in descending order and return the first
        highest_file = sorted(files, reverse=True)[0]
        filename = directory + "/" + highest_file
        return filename

    def create_dirs(self, dirs_to_create):
        # Create directories if they don't exist
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            if os.path.isdir(dir_path):
                print(f"Created or already exists: {dir_path}")
            else:
                print(f"Failed to create: {dir_path}")

    def _extract_part_timestamp(self, filename):
        pattern = r"^(part_\d+_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})"
        match = re.match(pattern, filename)
        return match.group(1) if match else None

    def add_scores_to_df(self, scores, directory, prefix, extension):
        """
        Add score and grade to CSV.
        """

        filename = self.get_highest_sorted_file(directory, prefix, extension)
        df = pd.read_csv(filename, index_col=None)

        try:
            # Prepare score columns
            score_columns = {
                name: [round(val, 6) if isinstance(val, float) else val]
                for name, val in scores.items()
            }
            new_cols_df = pd.DataFrame(score_columns)

            # Split df into left (before index 1) and right (from index 1 onward)
            left = df.iloc[:, :1]
            right = df.iloc[:, 1:]

            # Concatenate in correct order: left + new columns + right
            df = pd.concat([left, new_cols_df, right], axis=1)

            df.to_csv(filename, index=False)
        except Exception as e:
            print("Error adding scores to DataFrame:", e)

        return df

    def _combine_csv_files(self, directory, prefix, extension=".csv"):
        search_pattern = os.path.join(directory, f"{prefix}*{extension}")
        csv_files = glob.glob(search_pattern)

        if not csv_files:
            print("No matching CSV files found.")
            return None

        df_combined = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        print(f"Combined {len(csv_files)} files.")
        return df_combined

    def view_metrics(self, data_directory, plots_directory, prefix, sort_cols):
        df = self._combine_csv_files(data_directory, prefix)
        df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
        df = df.round({col: 6 for col in df.select_dtypes(include="float").columns})

        if df is not None:
            cols_to_view = [
                col
                for col in df.columns
                if col not in ["model_features", "model_architecture", "optimizer", "scheduler"]
            ]
            display(df[cols_to_view].head())

        print("\n============================== Filename ==============================")
        print(df["file_name"][0])
        print("\n============================== Model Architecture ==============================")
        print(df["model_architecture"][0])

        try:
            feature_list = ast.literal_eval(df["model_features"][0])
            print(f"\n============================== Model Features ==============================")
            for feature in feature_list:
                print(feature)
        except:
            print()

        print("\n============================== Optimizer ==============================")
        print(df["optimizer"][0])

        print("\n============================== Scheduler ==============================")
        print(df["scheduler"][0])

        print("\n============================== Sort Cols ==============================")
        for col in sort_cols:
            print(f"{col}: {df[col][0]}")

        file_prefix = df["file_name"][0]
        plots_filename = f"{plots_directory}/{file_prefix}_training_summary.png"

        display(Image(filename=plots_filename, width=1200, height=1200))
        return df
