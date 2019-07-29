import argparse
import os
import subprocess


def docker_launch(tutorial_name: str, build: bool, port: int) -> None:
    if not os.path.isdir(tutorial_name):
        raise ValueError(f"{tutorial_name} is not a valid tutorial")
    check_docker = subprocess.run(
        ["docker", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if check_docker.returncode:
        raise ValueError(
            f"Error calling Docker. Is it installed?\n"
            f"{str(check_docker.stdout, 'utf-8')}"
        )
    if build:
        arg = f"TUTORIAL={tutorial_name}"
        tag = f"--tag={tutorial_name}"
        subprocess.run(["docker", "build", "--build-arg", arg, tag, "."], check=True)
    port_config = f"{port}:8888"
    tag = f"{tutorial_name}:latest"
    check_run = subprocess.run(["docker", "run", "-it", "-p", port_config, tag])
    if check_run.returncode:
        raise ValueError(
            "Error running container. If you haven't built it yet, "
            "try running this script with the --build flag."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("tutorial_name", help="Name of the tutorial (directory)")
    parser.add_argument(
        "--build", action="store_true", default=False, help="Build the Docker image?"
    )
    parser.add_argument("--port", type=int, default=8888, help="Jupyter port for host")
    args = parser.parse_args()
    docker_launch(**vars(args))
