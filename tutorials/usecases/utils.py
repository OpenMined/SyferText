from tqdm import tqdm
import urllib.request as request
import os

def download_dataset(dataset_name: str, urls: list, root_path: str):
    """Download the dataset files from a list of urls.

    Args:
        dataset_name (str): The name of the dataset.
        urls (list of str): A list of urls to download.
        root_path (str): The path to the folder in which to download the dataset files.
    """

    # Intialize the totla size of the language model
    dataset_size = 0

    # chunk size during download (in bytes)
    chunk_size = 1024

    print(f"Preparing to download dataset: `{dataset_name}` ...")

    # Get the total size of all files combined
    for file_url in urls:
        file_info = request.urlopen(file_url).info()
        dataset_size += int(file_info["Content-Length"])

    # Initialize the progress bar object of tqdm
    prog_bar = tqdm(total=dataset_size, unit="B", unit_scale=True, desc=dataset_name)

    # start download
    for file_url in urls:

        # Create the file path
        file_name = os.path.basename(file_url)

        file_path = os.path.join(root_path, file_name)

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
