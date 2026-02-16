import gdown
import requests
from tqdm import tqdm


def download_file_from_google_drive(file_id, filename):
    gdown.download(file_id, filename, quiet=False, fuzzy=True)


def download_file_from_url(url: str, destination: str):
    """Download a file from a direct URL with a progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get total size in bytes
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 8192

    with open(destination, "wb") as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=destination
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    return destination
