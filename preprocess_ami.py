import argparse
import io
import requests
import pathlib
import shutil
import tempfile
import tarfile


EXPECTED_SIZE = 4267682
URL = "http://www.cs.columbia.edu/~kedzie/ami.v2.tar.gz"
CHUNK_SIZE = 32768

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, required=True,
        help="Parent directory for writing ami data directory.")
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
            print()
            raise Exception(
                "Download failed! "
                "Actual file size differs from expected file size!")

        print("Extracting data... ", end="", flush=True)
        f.seek(0)
        with tarfile.open(fileobj=f, mode='r:gz') as tf:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tf, args.data_dir)
        print(" done!")

if __name__ == "__main__":
    main()
