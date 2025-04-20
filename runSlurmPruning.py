def slurm_executor(
    user: str,
    host: str,
    remote_job_dir: str,
    account: str,
    partition: str,
    nodes: int,
    devices: int,
    time: str = "01:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    retries: int = 0,
) -> run.SlurmExecutor:
    if not (user and host and remote_job_dir and account and partition and nodes and devices):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes, and devices args for using this function."
        )

    mounts = []
    # Custom mounts are defined here.
    if custom_mounts:
        mounts.extend(custom_mounts)

    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    # This defines the slurm executor.
    # We connect to the executor via the tunnel defined by user, host, and remote_job_dir.
    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir, # This is where the results of the run will be stored by default.
            # identity="/path/to/identity/file" OPTIONAL: Provide path to the private key that can be used to establish the SSH connection without entering your password.
        ),
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,
        mem="0",
        exclusive=True,
        gres="gpu:8",
        packager=run.Packager(),
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor

def run_pretraining_with_slurm():
    recipe = configure_recipe(nodes=1, gpus_per_node=8)
    executor = slurm_executor(
        user="", # TODO: Set the username you want to use
        host="", # TODO: Set the host of your cluster
        remote_job_dir="", # TODO: Set the directory on the cluster where you want to save results
        account="", # TODO: Set the account for your cluster
        partition="", # TODO: Set the partition for your cluster
        container_image="", # TODO: Set the container image you want to use for your job
        # container_mounts=[], TODO: Set any custom mounts
        # custom_env_vars={}, TODO: Set any custom env vars
        nodes=recipe.trainer.num_nodes,
        devices=recipe.trainer.devices,
    )

    run.run(recipe, executor=executor, detach=True, name="nemotron3_4b_pretraining")

if __name__ == "__main__":
    run_pretraining_with_slurm()