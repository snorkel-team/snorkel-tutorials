import argparse
import errno
import os
import socket
import subprocess


def check_docker() -> None:
    try:
        subprocess.run(["docker", "--version"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise ValueError("Error calling Docker. Is it installed?")


def build_image(tutorial_name: str) -> None:
    arg = f"TUTORIAL={tutorial_name}"
    tag = f"--tag={tutorial_name}"
    subprocess.run(["docker", "build", "--build-arg", arg, tag, "."], check=True)


def check_port(port: int) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            raise ValueError(f"Port {port} is already in use")
        else:
            raise e


def run_image(tutorial_name: str, port: int) -> None:
    tag = f"{tutorial_name}:latest"
    p_cfg = f"{port}:{port}"
    p_arg = f"--port={port}"
    check_run = subprocess.run(["docker", "run", "-it", "-p", p_cfg, tag, p_arg])
    if check_run.returncode:
        raise ValueError(
            "Error running container. If you haven't built it yet, "
            "try running this script with the --build flag."
        )


def docker_launch(tutorial_name: str, build: bool, port: int) -> None:
    if not os.path.isdir(tutorial_name):
        raise ValueError(f"{tutorial_name} is not a valid tutorial")
    check_docker()
    if build:
        build_image(tutorial_name)
    check_port(port)
    run_image(tutorial_name, port)


if __name__ == "__main__":
    desc = "Build and run Docker images for Snorkel Tutorials."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("tutorial_name", help="Name of the tutorial (directory)")
    parser.add_argument(
        "--build", action="store_true", default=False, help="Build the Docker image?"
    )
    parser.add_argument("--port", type=int, default=8888, help="Jupyter port for host")
    args = parser.parse_args()
    docker_launch(**vars(args))
