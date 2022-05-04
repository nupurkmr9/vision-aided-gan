import hashlib
import requests
import glob
import os


def open_url(url, cache_dir=os.path.join(os.environ['HOME'], '.cache')):
    """return the local path in $USER/.cache for the given downloaded file from given url"""

    name = url.split('/')[-1]
    # check cache for already downloaded file.
    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
    if len(cache_files) == 1:
        return cache_files[0]

    print(f'Downloading {url}')
    url = requests.get(url)
    if len(url.content) == 0:
        raise IOError("No data received")

    url_data = url.content
    # Save to cache.
    cache_file = os.path.join(cache_dir, url_md5 + "_" + name)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "wb") as f:
        f.write(url_data)

    return cache_file
