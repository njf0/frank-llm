from kubejobs.jobs import KueueQueue
from kubejobs.pods import KubernetesPod
from kubernetes import client, config


def check_if_completed(job_name: str, namespace: str = 'informatics') -> bool:
    # Load the kube config
    config.load_kube_config()

    # Create an instance of the API class
    api = client.BatchV1Api()

    job_exists = False
    is_completed = True

    # Check if the job exists in the specified namespace
    jobs = api.list_namespaced_job(namespace)
    if job_name in {job.metadata.name for job in jobs.items}:
        job_exists = True

    if job_exists is True:
        job = api.read_namespaced_job(job_name, namespace)
        is_completed = False

        # Check the status conditions
        if job.status.conditions:
            for condition in job.status.conditions:
                if condition.type == 'Complete' and condition.status == 'True':
                    is_completed = True
                elif condition.type == 'Failed' and condition.status == 'True':
                    print(f'Job {job_name} has failed.')
        else:
            print(f'Job {job_name} still running or status is unknown.')

        if is_completed:
            api_res = api.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                body=client.V1DeleteOptions(propagation_policy='Foreground'),
            )
            print(f"Job '{job_name}' deleted. Status: {api_res.status}")
    return is_completed


# def send_message_command(cfg: Config):
#     # webhook - load from env
#     config.load_kube_config()
#     v1 = client.CoreV1Api()

#     secret_name = cfg.launch.env_vars["SLACK_WEBHOOK"]["secret_name"]
#     secret_key = cfg.launch.env_vars["SLACK_WEBHOOK"]["key"]

#     secret = v1.read_namespaced_secret(secret_name, "informatics").data
#     webhook = base64.b64decode(secret[secret_key]).decode("utf-8")
#     return (
#         """curl -X POST -H 'Content-type: application/json' --data '{"text":"Job started in '"$POD_NAME"'"}' """
#         + webhook
#         + " ; "
#     )


# def export_env_vars(cfg: Config):
#     cmd = ""
#     for key in cfg.launch.env_vars.keys():
#         cmd += f" export {key}=${key} &&"
#     cmd = cmd.strip(" &&") + " ; "
#     return cmd


# @hydra.main(config_path="configs", config_name="base", version_base=None)
def main():
    # cfg = Config(**dict(cfg))
    job_name = 'njf'
    # is_completed = check_if_completed(job_name, namespace="informatics")

    # if is_completed is True:
    print(f"Job '{job_name}' is completed. Launching a new job.")

    # TODO: make this interactive mode or not
    # if cfg.launch.interactive:
    command = 'while true; do sleep 60; done;'
    # else:
    #     plancraft_cfg = dict(cfg)["plancraft"]
    #     command = cfg.launch.command
    #     for key, value in plancraft_cfg.items():
    #         command += f" ++plancraft.{key}={value}"
    print(f'Command: {command}')

    # Create a Kubernetes Job with a name, container image, and command
    print(f'Creating job for: {command}')
    job = KubernetesPod(
        name=job_name,
        cpu_request=16,
        ram_request='112Gi',
        image='docker.io/goggledmapping0p/frank-llm:latest',
        gpu_type='nvidia.com/gpu',
        gpu_limit=1,
        gpu_product='NVIDIA-H100-80GB-HBM3',
        # backoff_limit=0,
        command=['/bin/bash', '-c', '--'],
        args=[command],
        user_email='s2242625@ed.ac.uk',
        namespace='informatics',
        kueue_queue_name=KueueQueue.INFORMATICS,
        volume_mounts={'nfs': {'mountPath': '/nfs', 'server': '10.24.1.255', 'path': '/'}},
    )

    job_yaml = job.generate_yaml()
    print(job_yaml)

    # Run the Job on the Kubernetes cluster
    job.run()
    # else:
    # print(f"Job '{job_name}' is still running.")


if __name__ == '__main__':
    main()
