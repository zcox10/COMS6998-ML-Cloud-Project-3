from kfp import dsl
from kfp import compiler
from kfp.client import Client
from datetime import datetime, timezone
import time
import logging
from typing import Dict


@dsl.component(base_image="gcr.io/zsc-personal/ml-cloud-pipeline:latest")
def data_preparation(
    num_tests: int, controller_params: dict, dynamics_params: dict, gcs_file_paths: Dict[str, str]
) -> Dict[str, str]:

    # imports
    import logging

    from src.data_pipeline import DataPipeline
    from src.utils.utils import Utils

    # enable logging
    Utils().configure_component_logging(log_level=logging.INFO)

    # run data pipeline
    return DataPipeline().run_data_pipeline(
        num_tests=num_tests,
        controller_params=controller_params,
        dynamics_params=dynamics_params,
        gcs_file_paths=gcs_file_paths,
    )


@dsl.component(base_image="gcr.io/zsc-personal/ml-cloud-pipeline:latest")
def train_model(gcs_file_paths: Dict[str, str]) -> Dict[str, str]:

    # imports
    import logging
    import torch

    from src.model_trainer import ModelTrainer
    from src.utils.utils import Utils

    # enable logging
    Utils().configure_component_logging(log_level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ModelTrainer(device=device).train(show_plot=False, gcs_file_paths=gcs_file_paths)


# define Kubeflow pipeline
@dsl.pipeline(name="ml-cloud-pipeline")
def pipeline():
    # Initialize arguments
    gcs_file_paths = {
        "bucket_name": "ml-cloud-kubeflow-pipeline-data",
        "model_path": "models",
        "eval_plot_path": "eval/plots",
        "eval_data_path": "eval/data",
        "training_data_path": "training_data",
    }
    controller_params = {
        "control_horizon": 1,
        "prediction_horizon": 1,
        "num_candidates": 50,
        "num_iterations": 10,
        "velocity_weight": 0.02,
        "distance_weight": 2.5,
        "data_collection": True,  # True for data collection; False for testing
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
    global_cache = False

    # Run data pipline component
    data_pipeline_task = data_preparation(
        num_tests=1,
        controller_params=controller_params,
        dynamics_params=dynamics_params,
        gcs_file_paths=gcs_file_paths,
    )
    data_pipeline_task.set_caching_options(global_cache)

    # Run model training component
    model_train_task = train_model(gcs_file_paths=gcs_file_paths).after(data_pipeline_task)
    model_train_task.set_caching_options(global_cache)


class RunPipeline:
    def __init__(self, host):
        self.client = Client(host=host)

    # compile pipeline to YAML
    def compile_to_yaml(self, pipeline_func, pipeline_package_path):
        compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=pipeline_package_path)
        logging.info(f"Pipeline compiled to {pipeline_package_path}")

    def get_latest_version(self, pipeline_id):
        pipeline_versions = self.client.list_pipeline_versions(
            pipeline_id=pipeline_id, page_size=100
        ).pipeline_versions

        max_version_timestamp = datetime.min.replace(tzinfo=timezone.utc)
        max_version_name = None
        for version in pipeline_versions:
            if version.created_at > max_version_timestamp:
                max_version_timestamp = version.created_at
                max_version_name = version.display_name

        return max_version_name

    def get_version_name(self, pipeline_id, incrementor):
        max_version_name = self.get_latest_version(pipeline_id)
        version_split = max_version_name.split(".")
        major_version = int(version_split[0])
        minor_version = int(version_split[1])

        if incrementor == "major":
            major_version += 1
        else:
            minor_version += 1

        return str(major_version) + "." + str(minor_version)

    def upload_pipeline(self, pipeline_name, pipeline_package_path, incrementor):

        pipeline_id = self.client.get_pipeline_id(pipeline_name)

        # Upload as new version
        if pipeline_id:
            logging.debug("Found pipeline_id, update version")

            version_name = self.get_version_name(pipeline_id, incrementor)
            pipeline_version = self.client.upload_pipeline_version(
                pipeline_package_path=pipeline_package_path,
                pipeline_version_name=version_name,
                pipeline_id=pipeline_id,
            )
            logging.info(
                f"Uploaded pipeline: {pipeline_name} version: {pipeline_version.display_name}"
            )

        # Upload as new pipeline
        else:
            logging.debug("New pipeline")
            pipeline = self.client.upload_pipeline(
                pipeline_package_path, pipeline_name=pipeline_name
            )
            time.sleep(1.1)

            pipeline_version = self.client.upload_pipeline_version(
                pipeline_package_path=pipeline_package_path,
                pipeline_version_name="1.0",
                pipeline_id=pipeline.pipeline_id,
            )
            logging.info(
                f"Uploaded pipeline: {pipeline_name} version: {pipeline_version.display_name}"
            )

        return pipeline_version.pipeline_id, pipeline_version.pipeline_version_id

    def get_or_create_experiment(self, experiment_name):
        experiment_id = self.client.get_pipeline_id(experiment_name)
        if not experiment_id:
            experiment_id = self.client.create_experiment(name=experiment_name).experiment_id
        return experiment_id

    def run_kubeflow_pipeline(self, job_name, experiment_name, pipeline_id, pipeline_version_id):
        experiment_id = self.get_or_create_experiment(experiment_name)
        run = self.client.run_pipeline(
            job_name=job_name,
            experiment_id=experiment_id,
            pipeline_id=pipeline_id,
            version_id=pipeline_version_id,
        )
        logging.info(f"Pipeline submitted. Run ID: {run.run_id}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="\n%(levelname)s: %(message)s\n")

    # constants
    pipeline_package_path = "./yaml/ml_cloud_pipeline.yaml"
    pipeline_name = "ml-cloud-pipeline"
    experiment_name = f"ml-cloud-data-experiment"
    host = "http://localhost:8080"
    service_account = "zsc-service-account@zsc-personal.iam.gserviceaccount.com"

    # Job name versionion
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"ml-cloud-data-run-{timestamp}"

    r = RunPipeline(host=host)
    r.compile_to_yaml(pipeline, pipeline_package_path)
    pipeline_id, pipeline_version_id = r.upload_pipeline(
        pipeline_name, pipeline_package_path, incrementor="minor"
    )
    r.run_kubeflow_pipeline(job_name, experiment_name, pipeline_id, pipeline_version_id)
