import argparse
import os
import subprocess
from typing import List


def get_modified_paths(travis_strict: bool) -> List[str]:
    base_branch = os.environ.get("$TRAVIS_BRANCH")
    if base_branch is None:
        if travis_strict:
            raise ValueError("No environment variable $TRAVIS_BRANCH")
        base_branch = "master"
    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", base_branch], stdout=subprocess.PIPE
    )
    cp = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", merge_base.stdout.rstrip()],
        stdout=subprocess.PIPE,
    )
    return [str(s, "utf-8") for s in cp.stdout.splitlines()]


def get_default_environments() -> List[str]:
    cp = subprocess.run(["tox", "-l"], stdout=subprocess.PIPE)
    return [str(s, "utf-8") for s in cp.stdout.splitlines()]


def get_changed_tox_envs(all_envs: bool, travis_strict: bool, plan: bool) -> None:
    # Check we're in the right place, otherwise git paths are messed up
    if os.path.split(os.getcwd())[1] != "snorkel-tutorials":
        raise ValueError("Execute this script from the snorkel-tutorials directory")
    # If we passed in --all flag, just run all environments
    default_environments = get_default_environments()
    if all_envs:
        if plan:
            print("Running all environments")
        print(",".join(default_environments))
        return
    # Find paths modified in patch
    modified_paths = get_modified_paths(travis_strict)
    if plan:
        print(f"Modified paths: {','.join(modified_paths)}")
    # Find unique snorkel-tutorial subdirectories affected by patch
    unique_directories = set()
    for p in modified_paths:
        splits = p.split("/")
        unique_directories.add("." if len(splits) == 1 else splits[0])
    # If we only have one and it's a valid tox environment, run that one
    if len(set(default_environments).difference(unique_directories)) == 1:
        changed_env = unique_directories.pop()
        if plan:
            print(f"Single changed tutorial directory: {changed_env}")
        print(changed_env)
    # Otherwise, run all environments
    else:
        if plan:
            print(
                f"No single changed tutorial directory [{unique_directories}], "
                "reverting to all environments"
            )
        print(",".join(default_environments))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--travis-strict",
        action="store_true",
        default=False,
        help="Fail if not in Travis?",
    )
    parser.add_argument(
        "--plan", action="store_true", default=False, help="Describe plan only?"
    )
    parser.add_argument(
        "--all",
        dest="all_envs",
        action="store_true",
        default=False,
        help="Run all environments?",
    )
    args = parser.parse_args()
    get_changed_tox_envs(**vars(args))
