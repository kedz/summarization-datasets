import rouge_papier
from duc_preprocess import duc2001
from duc_preprocess import duc2002

import argparse
import pathlib
import json
import re
import random
from multiprocessing import Pool, cpu_count

def validate_parent_dir(path):
    par = os.path.dirname(path)
    validate_dir(par)

def validate_dir(path):
    if path != "" and not os.path.exists(path):
        os.makedirs(path)

def get_labels(example, targets):
    summary_texts = []
    for tgt in targets:
        summary_texts.append(
            "\n".join([sent["text"] for sent in tgt["sentences"]]))

    input_texts = [sent["text"] if sent["word_count"] > 2 else "@@@@"
                   for sent in example["inputs"]]

    ranks, pairwise_ranks = rouge_papier.compute_extract(
        input_texts, summary_texts, mode="sequential", ngram=1,
        remove_stopwords=True, length=100)
    labels = [1 if r > 0 else 0 for r in ranks]
    labels = {"id": example["id"], "labels": labels}
    return labels

def convert_input(input):

    new_inputs = []
    for sent in input:
        new_inputs.append({
            "text": sent["text"],
            "tokens": sent["tokens"],
            "pos": sent["pos"],
            "ne": sent["ne"],
            "word_count": len(sent["text"].split())})
    id = "{}-{}".format(input[0]["docset_id"], input[0]["doc_id"])
    return {"id": id, "inputs": new_inputs}

def make_train_valid_data(duc2001_path, dest_dir, procs, valid_per=.15):
    orig_train_inputs = duc2001_path / "train" / "inputs"
    orig_train_targets = duc2001_path / "train" / "targets"
    orig_test_inputs = duc2001_path / "test" / "inputs"
    orig_test_targets = duc2001_path / "test" / "targets"
    total_examples = len([x for x in orig_train_inputs.glob("*.json")]) + \
        len([x for x in orig_test_inputs.glob("*.json")])

    train_and_valid_data = []

    for input_path in orig_train_inputs.glob("*.json"):
        input = convert_input(json.loads(input_path.read_text()))
        target_path = orig_train_targets / re.sub(
            r"input", r"target", input_path.name)
        target = json.loads(target_path.read_text())
        train_and_valid_data.append((input, target))

    for input_path in orig_test_inputs.glob("*.json"):
        input = convert_input(json.loads(input_path.read_text()))
        target_path = orig_test_targets / re.sub(
            r"input", r"target", input_path.name)
        target = json.loads(target_path.read_text())
        train_and_valid_data.append((input, target))

    # Shuffle train and valid data in a repeatable way.
    train_and_valid_data.sort(key=lambda x: x[0]["id"])
    random.shuffle(train_and_valid_data)

    # Sort is stable so this will keep relatively shuffled but put 
    # inputs with multiple human references toward the end of the list.
    # We would prefer to have these in the validation set since they 
    # will give more reliable rouge scores.
    train_and_valid_data.sort(key=lambda x: len(x[1]))

    valid_size = int(total_examples * valid_per)
    train_data = train_and_valid_data[:-valid_size]
    valid_data = train_and_valid_data[-valid_size:]

    valid_inputs_path = dest_dir / "inputs" / "valid"
    valid_labels_path = dest_dir / "labels" / "valid"
    valid_abs_path = dest_dir / "human-abstracts" / "valid"

    write_data(
        valid_inputs_path, valid_labels_path, valid_abs_path,
        valid_data, procs)

    train_inputs_path = dest_dir / "inputs" / "train"
    train_labels_path = dest_dir / "labels" / "train"
    train_abs_path = dest_dir / "human-abstracts" / "train"

    write_data(
        train_inputs_path, train_labels_path, train_abs_path, 
        train_data, procs)

def make_test_data(duc2002_path, dest_dir, procs, valid_per=.15):
    orig_test_inputs = duc2002_path / "inputs"
    orig_test_targets = duc2002_path / "targets"
    test_data = []
    for input_path in orig_test_inputs.glob("*.json"):
        input = convert_input(json.loads(input_path.read_text()))
        target_path = orig_test_targets / re.sub(
            r"input", r"target", input_path.name)
        target = json.loads(target_path.read_text())
        test_data.append((input, target))

    test_inputs_path = dest_dir / "inputs" / "test"
    test_labels_path = dest_dir / "labels" / "test"
    test_abs_path = dest_dir / "human-abstracts" / "test"

    write_data(
        test_inputs_path, test_labels_path, test_abs_path, 
        test_data, procs)

def get_labels_worker(args):
    example, targets, inputs_dir, labels_dir, abs_dir = args
    labels = get_labels(example, targets)
           
    input_path = inputs_dir / "{}.json".format(example["id"])
    input_path.write_text(json.dumps(example))
    labels_path = labels_dir / "{}.json".format(example["id"])
    labels_path.write_text(json.dumps(labels))
 
    for target in targets:
        target_path = abs_dir / "{}.{}.txt".format(
            example["id"], target["summarizer"].lower())
        target_path.write_text(
            "\n".join([s["text"] for s in target["sentences"]]))

def write_data(inputs_path, labels_path, abs_path, data, procs):
    inputs_path.mkdir(exist_ok=True, parents=True)
    labels_path.mkdir(exist_ok=True, parents=True)
    abs_path.mkdir(exist_ok=True, parents=True)

    data = [[x[0], x[1], inputs_path, labels_path, abs_path] for x in data]

    total_examples = len(data)
    pool = Pool(procs)
    result_iter = enumerate(pool.imap(get_labels_worker, data), 1)

    for i, _ in result_iter:
        print(
            "{}/{}".format(i, total_examples),
            end="\r" if i < total_examples else "\n",
            flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duc2001", type=pathlib.Path, required=True,
        help="Path to DUC2001_Summarization_Documents.tgz from NIST")
    parser.add_argument(
        "--duc2002-documents", type=pathlib.Path, required=True,
        help="Path to DUC2002_Summarization_Documents.tgz from NIST")
    parser.add_argument(
        "--duc2002-summaries", type=pathlib.Path, required=True,
        help="Path to DUC2002_test_data.tar.gz from script.")
    parser.add_argument(
        "--data-dir", type=pathlib.Path, required=True,
        help="Path to data directory to write duc-sds data.")
    parser.add_argument(
        "--seed", type=int, default=43929524)
    parser.add_argument("--procs", type=int, required=False, default=None)
    args = parser.parse_args()
    random.seed(args.seed)

    if args.procs is None:
        args.procs = min(cpu_count(), 16)

    data_dir = args.data_dir / "duc-sds"
    data_dir.mkdir(exist_ok=True, parents=True)

    duc2001_dir = data_dir / "duc2001"
    duc2002_dir = data_dir / "duc2002"

    duc2001.preprocess_sds(str(duc2001_dir), nist_data_path=str(args.duc2001))
    duc2002.preprocess_sds(
        str(duc2002_dir),
        nist_document_data_path=str(args.duc2002_documents),
        nist_summary_data_path=str(args.duc2002_summaries))
    make_train_valid_data(duc2001_dir, data_dir, args.procs)
    make_test_data(duc2002_dir, data_dir, args.procs)

if __name__ == "__main__":
    main()
