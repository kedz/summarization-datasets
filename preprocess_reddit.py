import argparse
import random
import requests
import io
import pathlib
import tarfile
import re
import spacy
import json
from multiprocessing import Pool, current_process
import rouge_papier


url = "http://www.cs.columbia.edu/~ouyangj/aligned-summarization-data/" + \
      "aligned-summarization-data.tar.gz"

def download_data():
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

def write_abstracts(story_ids, tar, output_dir): 

    output_dir.mkdir(parents=True, exist_ok=True)

    for name in tar.getnames():
        path = pathlib.Path(name)
        m = re.match(r"\./eacl_sample_full/annotator_(.+?)/abstractive/", name)

        if not m:
            continue
        if path.suffix != ".abstractive":
            continue
        if path.stem not in story_ids:
            continue

        story_id = path.stem
        annotator = m.groups()[0]
        output_file = output_dir / "{}.{}.txt".format(story_id, annotator)
        with tar.extractfile(name) as fp:
            output_file.write_bytes(fp.read())

def write_extracts(story_ids, tar, output_dir): 

    output_dir.mkdir(parents=True, exist_ok=True)

    for name in tar.getnames():
        path = pathlib.Path(name)
        m = re.match(r"\./eacl_sample_full/annotator_(.+?)/extractive/", name)

        if not m:
            continue
        if path.suffix != ".majority":
            continue
        if path.stem not in story_ids:
            continue

        story_id = path.stem
        annotator = m.groups()[0]
        output_file = output_dir / "{}.{}.txt".format(story_id, annotator)
        with tar.extractfile(name) as fp:
            output_file.write_bytes(fp.read())

def process_raw_input(text, nlp):
    inputs = []
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", r" ", line.strip())
        line = re.sub(r"\*\*\*\*+", ' ', line)
        lines.append(line)

    for doc in nlp.pipe(lines):
        for sent in doc.sents:
            text = sent.text.strip()
            tokens = [w.text.strip().lower() for w in sent]
            tc = len([w for w in tokens if len(w) > 0])
            if tc == 0:
                continue
            pos = [w.pos_ for w in sent]
            word_count = len(text.split())
            inputs.append({
                "text": text,
                "tokens": tokens,
                "pos": pos,
                "word_count": word_count})
    for i, input in enumerate(inputs, 1):
        input["sentence_id"] = i
    return inputs

def get_labels(example, extract_paths):
    summary_texts = []
    for path in extract_paths:
        with open(path, "r") as fp:
            summary_texts.append(fp.read())
    input_texts = [input["text"] if input["word_count"] > 2 else "@@@@@"
                   for input in example["inputs"]]
    ranks, pairwise_ranks = rouge_papier.compute_extract(
        input_texts, summary_texts, mode="sequential", ngram=1,
        remove_stopwords=True, length=75)
    labels = [1 if r > 0 else 0 for r in ranks]
    return labels

def init_worker(l_data_dir):
    global data_dir
    data_dir = l_data_dir
    global nlp
    nlp = spacy.load('en', parser=False)
    print("{}: Ready!".format(current_process().name))

def get_summary_paths(data_dir, story_id):
    paths = []
    for path in data_dir.glob("*"):
        if story_id == path.stem.rsplit(".", 1)[0]:
            paths.append(path) 
    return paths

def worker(args):
    story_id, raw_text, inputs_dir, ext_labels_dir, abs_labels_dir, part = args
    global data_dir
    global nlp

    inputs = process_raw_input(raw_text, nlp)    
    example = {"id": story_id, "inputs": inputs}
    inputs_path = inputs_dir / "{}.json".format(story_id)
    inputs_path.write_text(json.dumps(example))

    abs_paths = get_summary_paths(
        data_dir / "human-abstracts" / part, story_id)
    ext_paths = get_summary_paths(
        data_dir / "human-extracts" / part, story_id)

    ext_labels = {"id": story_id, "labels": get_labels(example, ext_paths)}
    ext_labels_path = ext_labels_dir / "{}.json".format(story_id)
    ext_labels_path.write_text(json.dumps(ext_labels))

    abs_labels = {"id": story_id, "labels": get_labels(example, abs_paths)}
    abs_labels_path = abs_labels_dir / "{}.json".format(story_id)
    abs_labels_path.write_text(json.dumps(abs_labels))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=3242342, required=False)
    parser.add_argument("--num-procs", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    data = download_data()

    tar = tarfile.open(mode="r:gz", fileobj=data)
    story_ids = [pathlib.Path(name).stem for name in tar.getnames()
                 if name.startswith("./eacl_sample_full/narrative/")]
    story_ids.sort()
    random.shuffle(story_ids)

    valid_size = round(.05 * len(story_ids))
    test_size = round(.1 * len(story_ids))
    train_size = len(story_ids) - valid_size - test_size
    train_ids = set(story_ids[:train_size])
    valid_ids = set(story_ids[train_size:train_size + valid_size])
    test_ids = set(story_ids[train_size + valid_size:])

    assert len(train_ids) == train_size
    assert len(valid_ids) == valid_size
    assert len(test_ids) == test_size
    assert len(valid_ids.intersection(train_ids)) == 0
    assert len(valid_ids.intersection(test_ids)) == 0
    assert len(test_ids.intersection(train_ids)) == 0

    abstracts_dir = args.data_dir / "reddit" / "human-abstracts"
    train_abstracts_dir = abstracts_dir / "train"
    valid_abstracts_dir = abstracts_dir / "valid"
    test_abstracts_dir = abstracts_dir / "test"
    extracts_dir = args.data_dir / "reddit" / "human-extracts"
    train_extracts_dir = extracts_dir / "train"
    valid_extracts_dir = extracts_dir / "valid"
    test_extracts_dir = extracts_dir / "test"

    print("Writing train abstracts to: {}".format(train_abstracts_dir))
    write_abstracts(train_ids, tar, train_abstracts_dir)
    print("Writing valid abstracts to: {}".format(valid_abstracts_dir))
    write_abstracts(valid_ids, tar, valid_abstracts_dir)
    print("Writing test abstracts to: {}".format(test_abstracts_dir))
    write_abstracts(test_ids, tar, test_abstracts_dir)

    print("Writing train extracts to: {}".format(train_extracts_dir))
    write_extracts(train_ids, tar, train_extracts_dir)
    print("Writing valid extracts to: {}".format(valid_extracts_dir))
    write_extracts(valid_ids, tar, valid_extracts_dir)
    print("Writing test extracts to: {}".format(test_extracts_dir))
    write_extracts(test_ids, tar, test_extracts_dir)

    data_dir = args.data_dir / "reddit"
    pool = Pool(
        args.num_procs, initargs=[data_dir], initializer=init_worker)


    make_dataset(
        train_ids,
        tar,
        data_dir / "inputs" / "train",
        data_dir / "ext_labels" / "train",
        data_dir / "abs_labels" / "train",
        "train",
        pool)

    make_dataset(
        valid_ids,
        tar,
        data_dir / "inputs" / "valid",
        data_dir / "ext_labels" / "valid",
        data_dir / "abs_labels" / "valid",
        "valid",
        pool)

    make_dataset(
        test_ids,
        tar,
        data_dir / "inputs" / "test",
        data_dir / "ext_labels" / "test",
        data_dir / "abs_labels" / "test",
        "test",
        pool)

def make_dataset(story_ids, tar, inputs_dir, ext_labels_dir, 
                 abs_labels_dir, part, pool):

    inputs_dir.mkdir(exist_ok=True, parents=True)
    ext_labels_dir.mkdir(exist_ok=True, parents=True)
    abs_labels_dir.mkdir(exist_ok=True, parents=True)

    def story_iter():
        for story_id in story_ids:
            name = "./eacl_sample_full/narrative/{}.story".format(story_id)
            with tar.extractfile(name) as fp:
                yield (story_id, fp.read().decode("utf8"),
                       inputs_dir, ext_labels_dir, abs_labels_dir, part)

    for i, _ in enumerate(pool.imap(worker, story_iter()), 1):
        print(
            "{}/{}".format(i, len(story_ids)), 
            end="\r" if i < len(story_ids) else "\n", 
            flush=True)

if __name__ == "__main__":
    main()
