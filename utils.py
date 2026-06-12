import os
import bz2
import shutil
import requests

from tqdm import tqdm

# Function to download files with progress bar
def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

# Function to extract bz2 files
def extract_bz2(filename):
    print(f"Extracting {filename}...")
    with bz2.BZ2File(filename, 'rb') as source, open(filename[:-4], 'wb') as dest:
        shutil.copyfileobj(source, dest)
    os.remove(filename)  # Remove the compressed file
    print(f"Extracted {filename}")