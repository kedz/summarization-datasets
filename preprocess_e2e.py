import argparse
import pathlib
import csv
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import ujson as json
import requests
import io
import zipfile
import shutil
import tempfile


def download_data():
    url = "https://github.com/tuetschek/e2e-dataset/releases/download/" + \
          "v1.0.0/e2e-dataset.zip"
    session = requests.Session()
    response = session.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", "-1"))

    progress_template = "{:" + str(len(str(file_size))) + "d} / {:d}\r"
    chunk_size = 32768
    chunks = []
    bytes_read = 0
    for chunk in response.iter_content(chunk_size):
        if chunk:
            chunks.append(chunk)
            bytes_read += len(chunk)
            print(progress_template.format(bytes_read, file_size),
                  end="" if bytes_read < file_size else "\n", flush=True)
    data = b''.join(chunks)
    return io.BytesIO(data)

def preprocess_text(raw_text):
    tokens = []
    for i, sent in enumerate(sent_tokenize(raw_text)):
        if i > 0:
            tokens.append("<SENT>")
        for token in word_tokenize(sent):
            tokens.append(token.lower())
    return {"text": raw_text, "tokens": {"tokens": tokens}}

def get_structured_data(line):
    data = {}
    for item in line.split(","):
        item = item.strip()
        match = re.match(r'(.+?)\[(.*?)\]', item) 
        assert match
        key, value = match.groups()
        data[key] = value
    fields = sorted(data.items(), key=lambda x: x[0])
    tokens = {"fields": [], "tokens": [], "positions": []}
    position = 1
    for i, (field, field_string) in enumerate(fields, 1):
        for token in word_tokenize(field_string):
            tokens["fields"].append(field) 
            tokens["tokens"].append(token.lower())
            tokens["positions"].append(str(position))
            position += 1

    return {"data": data, "tokens": tokens}

def preprocess_data(input_path, source_path, target_path):
    with input_path.open('r') as in_fp: 
        with source_path.open('w') as src_fp, target_path.open('w') as tgt_fp:
            csv_reader = csv.reader(in_fp, delimiter=',')
            next(csv_reader)

            for row in csv_reader:
                source = get_structured_data(row[0])
                src_fp.write(json.dumps(source))
                src_fp.write("\n")
                target = preprocess_text(row[1])
                tgt_fp.write(json.dumps(target))
                tgt_fp.write("\n")

def make_references(input_path, ref_source, ref_target):
    with input_path.open('r') as in_fp: 
        with ref_source.open("w") as src_fp, ref_target.open("w") as tgt_fp:
            csv_reader = csv.reader(in_fp, delimiter=',')
            next(csv_reader)

            cur_data = None
            cur_outputs = []
            for row in csv_reader:
                if cur_data is None:
                    cur_data = row[0]
                    cur_outputs.append(row[1])
                elif cur_data == row[0]:
                    cur_outputs.append(row[1])
                else:
                    src_fp.write(json.dumps(get_structured_data(cur_data)))
                    src_fp.write("\n")
                    tgt_fp.write("\n".join(cur_outputs))
                    tgt_fp.write("\n\n")
                    cur_data = row[0]
                    cur_outputs = [row[1]]  
                    
            src_fp.write(json.dumps(get_structured_data(cur_data)))
            tgt_fp.write("\n".join(cur_outputs))

def main():
    parser = argparse.ArgumentParser("Preprocess E2E generation datset.")
    parser.add_argument("output_dir", type=pathlib.Path)
    args = parser.parse_args()

    raw_data = download_data()
    tempdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(raw_data, mode='r') as zf:
            zf.extractall(tempdir)

        data_dir = pathlib.Path(tempdir) / "e2e-dataset"
        train_csv = data_dir / "trainset.csv"
        valid_csv = data_dir / "devset.csv"
        test_csv = data_dir / "testset_w_refs.csv"

        output_dir = args.output_dir / "e2e"
        output_dir.mkdir(parents=True, exist_ok=True)
        train_source = output_dir / "e2e.train.source.json"
        train_target = output_dir / "e2e.train.target.json"
        preprocess_data(train_csv, train_source, train_target)

        valid_source = output_dir / "e2e.valid.source.json"
        valid_target = output_dir / "e2e.valid.target.json"
        preprocess_data(valid_csv, valid_source, valid_target)

        valid_ref_source = output_dir / "e2e.valid.ref.source.json"
        valid_ref_target = output_dir / "e2e.valid.ref.target.txt"
        make_references(valid_csv, valid_ref_source, valid_ref_target)

        test_ref_source = output_dir / "e2e.test.ref.source.json"
        test_ref_target = output_dir / "e2e.test.ref.target.txt"
        make_references(test_csv, test_ref_source, test_ref_target)

    finally:
        try:
            shutil.rmtree(tempdir)
        except IOError:
            print('Failed to clean up temp dir {}'.format(tempdir))

if __name__ == "__main__":
    main()
