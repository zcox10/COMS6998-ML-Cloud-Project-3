from kfp import dsl
from kfp import compiler
from src.data_preparation import data_preparation_component


@dsl.pipeline(name="ml-cloud-pipeline")
def pipeline():
    data_task = data_preparation_component(
        num_tests=10,
        controller_params={
            "control_horizon": 1,
            "prediction_horizon": 1,
            "num_candidates": 50,
            "num_iterations": 10,
            "velocity_weight": 0.02,
            "distance_weight": 2.5,
            "data_collection": False,
        },
        dynamics_params={
            "num_links": 2,
            "link_mass": 0.1,
            "link_length": 1,
            "joint_viscous_friction": 0.1,
            "dt": 0.01,
        },
        output_path="/tmp/data.pkl",
    )


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=pipeline, package_path="ml_cloud_pipeline.yaml")
