import torch
import time
import os
import re
import glob
import ast
import numpy as np
import pandas as pd
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage
from matplotlib.figure import Figure
import tempfile
import torch.nn as nn
from IPython.display import Image, display


class GenericUtils:
    def __init__(self):
        self.client = storage.Client()

    def retrieve_latest_gcs_parquet_file(self, gcs_bucket_name, gcs_output_path):
        bucket = self.client.bucket(gcs_bucket_name)

        # List all blobs in the directory
        blobs = list(bucket.list_blobs(prefix=gcs_output_path))
        parquet_blobs = [b for b in blobs if b.name.endswith(".parquet")]

        if not parquet_blobs:
            raise FileNotFoundError("No parquet files found in the GCS path.")

        # Sort files based on filename assuming timestamp format
        latest_blob = sorted(parquet_blobs, key=lambda b: b.name, reverse=True)[0]

        # Download the latest file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_file:
            latest_blob.download_to_filename(temp_file.name)
            local_path = temp_file.name

        # Read parquet into a dictionary of numpy arrays
        table = pq.read_table(local_path).to_pydict()
        X = np.array(table["X"], dtype=np.float32)
        Y = np.array(table["Y"], dtype=np.float32)
        return {"X": X, "Y": Y}

    def _upload_file_to_gcs(
        self, local_path: str, gcs_bucket_name: str, gcs_output_path: str, save_filename: str
    ) -> str:
        # Upload to GCS
        gcs_uri = f"gs://{gcs_bucket_name}/{gcs_output_path}/{save_filename}"
        logging.info(f"Uploading to GCS: {gcs_uri}")

        client = storage.Client()
        bucket = client.bucket(gcs_bucket_name)
        blob = bucket.blob(f"{gcs_output_path}/{save_filename}")
        blob.upload_from_filename(local_path)

        os.remove(local_path)
        return gcs_uri

    def save_asset_to_gcs(self, asset, gcs_bucket_name, gcs_output_path, save_filename_prefix):

        # for PNG (plots)
        if isinstance(asset, Figure):
            filename = f"{save_filename_prefix}.png"
            local_path = f"/tmp/{filename}"
            asset.savefig(local_path, dpi=150)

        # for PyTorch modules
        elif isinstance(asset, nn.Module):
            filename = f"{save_filename_prefix}.pth"
            local_path = f"/tmp/{filename}"
            torch.save(asset.state_dict(), local_path)

        # for pandas dataframes
        elif isinstance(asset, pd.DataFrame):
            filename = f"{save_filename_prefix}.csv"
            local_path = f"/tmp/{filename}"
            asset.to_csv(local_path, index=False)

        # for parquet data files
        elif isinstance(asset, pa.Table):
            filename = f"{save_filename_prefix}.parquet"
            local_path = f"/tmp/{filename}"
            pq.write_table(asset, local_path)

        return self._upload_file_to_gcs(local_path, gcs_bucket_name, gcs_output_path, filename)

    def configure_component_logging(self, log_level):
        """
        Configure logging according to an associated log_level.
        """
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=log_level, format="\n%(levelname)s: %(message)s\n")

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
        logging.info(f"{message}: {minutes} min {seconds:.2f} sec")
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
                logging.info(f"Created or already exists: {dir_path}")
            else:
                logging.info(f"Failed to create: {dir_path}")

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
            logging.info("Error adding scores to DataFrame:", e)

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
