# How to use with Kubernetes cluster

## Building image with kaniko

Ensure you have a `Dockerfile` at the root of your repository.

### Register a secret with docker-hub credentials

Save a file as `config.json` with the following:

```json
{
    "auths": {
        "https://index.docker.io/v1/": {
            "auth": "base64 encode of username:password"
        }
    }
}
```

Then run the following command:

```bash
kubectl create secret generic regcred-goggledmapping0p --from-file=config.json=config.json --namespace=informatics
```

Replace `XXX` with your username and `your-namespace` with your namespace.

### Build Image on Kube

Create a file `kaniko.yaml` with the following:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kaniko
  labels:
    eidf/user: "s2242625" # Replace this with your EIDF username
spec:
  containers:
  - name: kaniko
    resources:
      requests:
        cpu: "500m"  # Requests 0.5 CPU cores
        memory: "1Gi"  # Requests 1 GiB of memory
      limits:
        cpu: "1"  # Limits to 1 CPU core
        memory: "2Gi"  # Limits to 2 GiB of memory
    image: gcr.io/kaniko-project/executor:latest
    args: ["--dockerfile=Dockerfile",
           "--context=git://github.com/goggledmapping0p/frank-llm.git#main", # Replace with your git repo - must be public
           "--destination=docker.io/goggledmapping0p/frank-llm:latest", # Replace with your docker hub image
           "--cache=true"]
    volumeMounts:
      - name: docker-config
        mountPath: /kaniko/.docker
  volumes:
    - name: docker-config
      secret:
        secretName: regcred-goggledmapping0p
  restartPolicy: Never
```

Then run the following command:

```bash
kubectl delete pod kaniko-test && kubectl apply -f build/kaniko-pod.yaml
```

This will create a pod that will build your image and push it to docker hub.

## Using the image

Once the image is built and pushed to the hub, you can use it in your kubernetes cluster.

### Adding local environment variables

Create unique secret with environment variables:

```bash
kubectl create secret generic s2234411-hf --from-literal=HF_TOKEN=hf_***
kubectl create secret generic s2234411-openai --from-literal=OPENAI_API_KEY=sk-***
kubectl create secret generic s2234411-wandb --from-literal=WANDB_API_KEY=***
# Optional: slack webhook to get notified when pod starts
kubectl create secret generic s2234411-slack-webhook --from-literal=SLACK_WEBHOOK=***
```

### Update the launch config in your run config

Example of my config `configs/text-env/base.yaml` that declares the environment variables and GPU limits for the pod:

```yaml
launch:
  gpu_limit: 1
  gpu_product: NVIDIA-A100-SXM4-80GB
  env_vars:
    HF_TOKEN:
      secret_name: s2234411-hf
      key: HF_TOKEN
    OPENAI_API_KEY:
      secret_name: s2234411-openai
      key: OPENAI_API_KEY
    WANDB_API_KEY:
      secret_name: s2234411-wandb
      key: WANDB_API_KEY
    SLACK_WEBHOOK:
      secret_name: s2234411-slack-webhook
      key: SLACK_WEBHOOK
```

### Deploy the pod

To deply use `kubejobs` library, and the provided `launch.py` script.

```bash
python launch.py ++launch.job-name=gautier-test-job ++launch.gpu-type=NVIDIA-A100-SXM4-80GB ++launch.gpu-limit=1
```

#### Interactive session

To get an interactive session:

```bash
python launch.py ++launch.job-name=gautier-test-job ++launch.gpu-type=NVIDIA-A100-SXM4-40GB ++launch.gpu-limit=1 ++launch.interactive=True
```

Once the pod is live, you can run the following command:

```bash
kubectl exec -it <job-name> -- /bin/bash
```

#### Monitoring

To monitor the pod, you can use the following command:

```bash
kubectl logs -f <job-name>
```
