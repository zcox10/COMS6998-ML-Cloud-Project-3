import kfp
from kfp.client import Client

# URL for your Kubeflow Pipelines endpoint
# Change this if you're not running in localhost
client = Client(host="http://<your-kubeflow-pipelines-endpoint>")

# 1. Upload the compiled pipeline YAML
pipeline_name = "robot-data-pipeline"
pipeline_package_path = "robot_pipeline.yaml"

pipeline = client.upload_pipeline(pipeline_package_path, pipeline_name=pipeline_name)

# 2. Create (or find) an experiment
experiment = client.create_experiment(name="robot-experiment")

# 3. Run the pipeline
run = client.run_pipeline(
    experiment_id=experiment.id,
    job_name="robot-data-run",
    pipeline_id=pipeline.id,
)

print(f"Pipeline submitted: {run.id}")
