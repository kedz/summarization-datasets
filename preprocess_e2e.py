import argparse
import pathlib
import csv
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import ujson as json


def preprocess_text(raw_text):
    tokens = []
    for i, sent in enumerate(sent_tokenize(raw_text)):
        if i > 0:
            tokens.append("<SENT>")
        for token in word_tokenize(sent):
            tokens.append(token.lower())
    return {"text": raw_text, "tokens": tokens}

def get_structured_data(line):
    data = {}
    for item in line.split(","):
        item = item.strip()
        match = re.match(r'(.+?)\[(.*?)\]', item) 
        assert match
        key, value = match.groups()
        data[key] = value
    fields = sorted(data.items(), key=lambda x: x[0])
    tokens = []
    position = 1
    for i, (field, field_string) in enumerate(fields, 1):
        for token in word_tokenize(field_string):
            tokens.append({"field": field, "token": token.lower(), 
                           "position": position})
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

def main():
    parser = argparse.ArgumentParser("Preprocess E2E generation datset.")
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    
    output_dir = args.output_dir / "e2e"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = args.input_dir / "trainset.csv"
    train_source = output_dir / "e2e.train.source.json"
    train_target = output_dir / "e2e.train.target.json"
    preprocess_data(train_csv, train_source, train_target)

    valid_csv = args.input_dir / "devset.csv"
    valid_source = output_dir / "e2e.valid.source.json"
    valid_target = output_dir / "e2e.valid.target.json"
    preprocess_data(valid_csv, valid_source, valid_target)

if __name__ == "__main__":
    main()
