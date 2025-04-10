# Installation Steps

## 1. Create a Kubernetes Cluster on GKE

These steps assume you have configured your GCP project and have authenticated gcloud on your local machine.  Run the following command to create a Kubernetes cluster:

```bash
gcloud container clusters create <cluster_name> \
  --zone us-east1-c \
  --machine-type=e2-standard-4 \
  --num-nodes=3 \
  --enable-autoscaling --min-nodes=1 --max-nodes=4 \
  --enable-ip-alias \
  --release-channel=regular \
  --workload-pool=$(gcloud config get-value project).svc.id.goog

```

This command will install a Standard Kubernetes cluster (non-Autopilot) and enable us to later install Kubeflow Pipelines to run an ML pipeline.

Next, authenticate to the cluster on your local machine with the following command:

```bash
gcloud container clusters get-credentials <your-cluster-name> --zone <your-cluster-zone>
```

**Helpful commands:**

- `kubectl get pods -A -w` views the status of all pods
- `kubectl get pods -n kubeflow -w` views the status of all Kubeflow pods
- `kubectl logs -n <namespace>` views the logs of a specific pod

## 2. Install Kubeflow Pipelines

Installing Kubeflow Pipelines will allow us to create an end-to-end ML pipeline for data preparation, model training, and model deployment. The instructions to install Kubeflow Pipelines are found [here](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/).

**Troubleshooting:**
When originally running the installation commands, the operation received a `Permission Denied` error when attempting to spin up the MySQL and MinIO pods.  The solution to this was to add an init container that would allow permission to write to these persistent volumes.  To successfully troubleshoot and add necessary permissions, first export the deployment YAML for MinIO or MySQL:

kubectl get deployment mysql -n kubeflow -o yaml \
  | kubectl neat > mysql-deployment.yaml

```bash
kubectl get deployment minio -n kubeflow -o yaml \
  | kubectl neat > minio-deployment.yaml
```

Then, under `spec.template.spec`, add `initContainers` to allow writing permissions for persistent volumes:

```bash
    initContainers:
    - name: fix-perms
      image: busybox
      command: ["sh", "-c", "chown -R 1000:1000 /export"]
      volumeMounts:
        - name: export
          mountPath: /export
```

Additionally, with MySQL, retrieve the deployment YAML with:

```bash
kubectl get deployment minio -n kubeflow -o yaml \
  | kubectl neat > minio-deployment.yaml
```

Then, apply the necessary permissions for MySQL to deploy successfully under `spec.template.spec`:

```bash
initContainers:
      - name: fix-mysql-perms
        image: busybox
        command: ["sh", "-c", "chown -R 1000:0 /var/lib/mysql"]
        volumeMounts:
          - name: mysql-persistent-storage
            mountPath: /var/lib/mysql
```

We can validate that everything is running properly by running:

Then, we can view the Kubeflow UI by running the following command: `kubectl get pods -A -w`.

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Then, visit `http://localhost:8080`. You should see no errors.
