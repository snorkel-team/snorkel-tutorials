from typing import Sequence
import os
import shutil
import urllib.request
import zipfile


def download_files(files: Sequence[str], data_url: str, directory: str):
    # Check that we are running from the right directory.
    if directory and os.path.split(os.getcwd())[1] != directory:
        raise Exception(f"Script must be run from {directory} directory")
    reload = any(not os.path.exists(os.path.join("data", filename)) for filename in files)
    if reload:
        if os.path.exists("data/"):
            shutil.rmtree("data/")
        os.mkdir("data")
        urllib.request.urlretrieve(data_url, "data.zip")
        with zipfile.ZipFile("data.zip", "r") as zip_ref:
            zip_ref.extractall("data")
        os.remove("data.zip")