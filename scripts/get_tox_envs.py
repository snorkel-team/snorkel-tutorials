import argparse
import os
import subprocess
from typing import List

EXTRA_ENVIRONMENTS = ["style"]
SKIP_EXT = [".md", ".txt"]


def get_modified_paths(no_travis_strict: bool) -> List[str]:
    # Call git diff --name-only HEAD $(git merge-base HEAD $TRAVIS_BRANCH)
    # to get paths affected by patch
    base_branch = os.environ.get("TRAVIS_BRANCH")
    if base_branch is None:
        if not no_travis_strict:
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
    # Call tox -l to get default environments
    cp = subprocess.run(["tox", "-l"], stdout=subprocess.PIPE)
    return [str(s, "utf-8") for s in cp.stdout.splitlines()]


def get_changed_tox_envs(all_envs: bool, no_travis_strict: bool, plan: bool) -> None:
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
    modified_paths = get_modified_paths(no_travis_strict)
    if plan:
        print(f"Modified paths: {','.join(modified_paths)}")
    # Find unique snorkel-tutorial subdirectories affected by patch
    unique_directories = set()
    for p in modified_paths:
        # Skip changed markdown / text files as they don't need a test env.
        if any(p.endswith(ext) for ext in SKIP_EXT):
            continue
        splits = p.split("/")
        # If there's a directory, parse it; otherwise, add placeholder "."
        unique_directories.add("." if len(splits) == 1 else splits[0])
    unique_defaults = [d for d in unique_directories if d in default_environments]
    # If all changed directories are among the defaults, then only run them
    # plus EXTRA_ENVIRONMENTS.
    if len(unique_defaults) == len(unique_directories):
        run_environments = unique_defaults + EXTRA_ENVIRONMENTS
        if plan:
            print(
                f"Changed tutorial directories: {unique_defaults}, "
                f"running environments: {run_environments}"
            )
        print(",".join(run_environments))
    # Otherwise, run all environments
    else:
        if plan:
            print(
                "Change in non-tutorial directory. "
                f"All changes: [{unique_directories}]. "
                "Running on all environments."
            )
        print(",".join(default_environments))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-travis-strict",
        action="store_true",
        default=False,
        help="Don't fail if not in Travis?",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        default=False,
        help="Print out plan for Travis execution?",
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
