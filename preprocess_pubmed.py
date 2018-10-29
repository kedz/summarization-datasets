import argparse
import io
import requests
import pathlib
import shutil
import tempfile
import tarfile


EXPECTED_SIZE = 320807981
URL = "http://www.cs.columbia.edu/~kedzie/pubmed.tar.gz"
CHUNK_SIZE = 32768

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, required=True,
        help="Parent directory for writing pubmed data directory.")
    args = parser.parse_args()
   
    args.data_dir.mkdir(exist_ok=True, parents=True) 

    session = requests.Session()
    response = session.get(URL, stream=True)

    with io.BytesIO() as f:
        size = 0
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                size += len(chunk)
            print(
                "[{:10d} of {:10d}]".format(size, EXPECTED_SIZE),
                end="\r" if size < EXPECTED_SIZE else "\n",
                flush=True)
        if size != EXPECTED_SIZE:
            raise Exception(
                "Download failed! "
                "Actual file size differs from expected file size!")

        print("Extracting data... ", end="", flush=True)
        f.seek(0)
        with tarfile.open(fileobj=f, mode='r:gz') as tf:
            tf.extractall(args.data_dir)
        print(" done!")

if __name__ == "__main__":
    main()
