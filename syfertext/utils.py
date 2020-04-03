import mmh3
import os
import logging
import urllib.request as request
from tqdm import tqdm
from pathlib import Path
import tempfile
import shutil

# Files to download for each language model
# TODO: Downloading language models should be handled
#       differently in later version of the library.
#       One good idea is to download language models as python packages
#       as done in spaCy

lang_model_files = dict()

lang_model_files["en_core_web_lg"] = [
    "https://github.com/AlanAboudib/syfertext_en_core_web_lg/blob/master/key2row?raw=true",
    "https://github.com/AlanAboudib/syfertext_en_core_web_lg/blob/master/vectors?raw=true",
    "https://github.com/AlanAboudib/syfertext_en_core_web_lg/blob/master/words?raw=true",
]


def hash_string(string):

    key = mmh3.hash64(string, signed=False, seed=1)[0]

    return key


def get_lang_model(model_name: str):
    """Downloads the specified language model `model_name` if not already done.

     Checks if the language folder named `model_name` is present. If not, it
     creates it and downloads the language model files  inside.

     Todo:
         This is an intial version to how language models are dealt
         with. it should be revisited later.
    """

    # Path to the folder containing language models
    data_path = os.path.join(str(Path.home()), "SyferText")

    # Do not download the language model if it is already done
    download_model = False

    # If the data folder does not exist yet, create it
    if not os.path.isdir(data_path):

        os.mkdir(data_path)

    # If the data folder does not contain the language model
    # folder, download the language model
    if model_name not in os.listdir(data_path):

        download_model = True

    if download_model:

        # download model files into a temporary path
        tmp_model_path = _download_model(model_name)

        # move temporary model directory to the models folder
        shutil.move(tmp_model_path, data_path)


def _download_model(model_name: str):
    """Download the language model files through HTTP.

    Args:
        model_name (str): The name of the language model.
    """

    # Intialize the total size of the language model
    model_size = 0

    # chunk size during download (in bytes)
    chunk_size = 1024

    print(f"Preparing to download language model `{model_name}` ...")

    # Get the total size of all files combined
    for file_url in lang_model_files[model_name]:
        file_info = request.urlopen(file_url).info()
        model_size += int(file_info["Content-Length"])

    # Initialize the progress bar object of tqdm
    prog_bar = tqdm(total=model_size, unit="B", unit_scale=True, desc=model_name)

    # get the temporary directory path
    tmp_data_path = tempfile.gettempdir()

    # create the temporary model path
    tmp_model_path = os.path.join(tmp_data_path, model_name)

    if not os.path.exists(tmp_model_path):
        os.mkdir(tmp_model_path)

    # start download
    for file_url in lang_model_files[model_name]:

        # Create the file path
        file_name = os.path.basename(file_url).split("?")[0]

        file_path = os.path.join(tmp_model_path, file_name)

        # Create the request object
        req = request.urlopen(file_url)

        with (open(file_path, "wb")) as f:

            while True:

                # Read a chunk of the file
                chunk = req.read(chunk_size)

                # If the chunck is not empty, write it to the file
                if chunk:
                    f.write(chunk)

                    # update tqdm's progress bar
                    prog_bar.update(chunk_size)

                else:
                    # Downloading the file finished
                    break
    prog_bar.close()
    
    return tmp_model_path

def normalize_slice(length, start, stop, step=None):

    assert step is None or step == 1, "Not a valid Slice"

    if start is None:
        start = 0

    elif start < 0:
        start += length

    start = min(length, max(0, start))

    if stop is None:
        stop = length

    elif stop < 0:
        stop += length

    stop = min(length, max(start, stop))

    assert start < stop, "Empty range"

    return start, stop