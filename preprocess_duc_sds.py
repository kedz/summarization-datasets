import argparse
import sys
import os
import json
import re
import random
from duc_preprocess import duc2001
from duc_preprocess import duc2002
import rouge_papier
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
    orig_train_inputs = os.path.join(duc2001_path, "train", "inputs")
    orig_train_targets = os.path.join(duc2001_path, "train", "targets")
    orig_test_inputs = os.path.join(duc2001_path, "test", "inputs")
    orig_test_targets = os.path.join(duc2001_path, "test", "targets")
    total_examples = len(os.listdir(orig_train_inputs)) + \
        len(os.listdir(orig_test_inputs))

    train_and_valid_data = []
    #train_and_valid_data = os.listdir(orig_train_inputs)
    #train_and_valid_data += os.listdir(orig_test_inputs)
    #train_and_valid_data.sort()

    for input_filename in train_and_valid_data:

        input_json = os.path.join(orig_train_inputs, input_filename)
        with open(input_json, "r") as inp_fp:
            input = json.loads(inp_fp.read())
            input = convert_input(input)

        target_json = os.path.join(
            orig_train_targets, re.sub(r"input", r"target", input_filename))
        with open(target_json, "r") as tgt_fp:
            target = json.loads(tgt_fp.read())

        train_and_valid_data.append((input, target))

    for input_filename in os.listdir(orig_test_inputs):

        input_json = os.path.join(orig_test_inputs, input_filename)
        with open(input_json, "r") as inp_fp:
            input = json.loads(inp_fp.read())
            input = convert_input(input)

        target_json = os.path.join(
            orig_test_targets, re.sub(r"input", r"target", input_filename))
        with open(target_json, "r") as tgt_fp:
            target = json.loads(tgt_fp.read())

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


    valid_inputs_path = os.path.join(
        dest_dir, "inputs", "duc-sds.inputs.valid.json")
    valid_labels_path = os.path.join(
        dest_dir, "labels", "duc-sds.labels.valid.json")
    valid_abs_path = os.path.join(
        dest_dir, "human-abstracts", "valid")

    write_data(
        valid_inputs_path, valid_labels_path, valid_abs_path, valid_data,
        procs)

    train_inputs_path = os.path.join(
        dest_dir, "inputs", "duc-sds.inputs.train.json")
    train_labels_path = os.path.join(
        dest_dir, "labels", "duc-sds.labels.train.json")
    train_abs_path = os.path.join(
        dest_dir, "human-abstracts", "train")

    write_data(
        train_inputs_path, train_labels_path, train_abs_path, train_data,
        procs)

def make_test_data(duc2002_path, dest_dir, procs, valid_per=.15):
    orig_test_inputs = os.path.join(duc2002_path, "inputs")
    orig_test_targets = os.path.join(duc2002_path, "targets")
    test_data = []
    for input_filename in os.listdir(orig_test_inputs):

        input_json = os.path.join(orig_test_inputs, input_filename)
        with open(input_json, "r") as inp_fp:
            input = json.loads(inp_fp.read())
            input = convert_input(input)

        target_json = os.path.join(
            orig_test_targets, re.sub(r"input", r"target", input_filename))
        with open(target_json, "r") as tgt_fp:
            target = json.loads(tgt_fp.read())

        test_data.append((input, target))
    test_data.sort(key=lambda x: x[0]["id"])

    test_inputs_path = os.path.join(
        dest_dir, "inputs", "duc-sds.inputs.test.json")
    test_labels_path = os.path.join(
        dest_dir, "labels", "duc-sds.labels.test.json")
    test_abs_path = os.path.join(
        dest_dir, "human-abstracts", "test")

    write_data(
        test_inputs_path, test_labels_path, test_abs_path, test_data, procs)

def get_labels_worker(args):
    example, targets = args
    labels = get_labels(example, targets)
    return example, targets, labels

def write_data(inputs_path, labels_path, abs_path, data, procs):
    validate_parent_dir(inputs_path)
    validate_parent_dir(labels_path)
    validate_dir(abs_path)
    total_examples = len(data)
    pool = Pool(procs)
    result_iter = enumerate(pool.imap(get_labels_worker, data), 1)

    with open(inputs_path, "w") as inp_fp, open(labels_path, "w") as lbl_fp:
        for i, (example, targets, labels) in result_iter:
            sys.stdout.write("{}/{}\r".format(i, total_examples))
            sys.stdout.flush()

            inp_fp.write(json.dumps(example))
            inp_fp.write("\n")
            lbl_fp.write(json.dumps(labels))
            lbl_fp.write("\n")
            for target in targets:
                tgt_path = os.path.join(
                    abs_path,
                    "{}.{}.txt".format(example["id"], target["summarizer"].lower()))
                with open(tgt_path, "w") as tgt_fp:
                    tgt_fp.write("\n".join([s["text"] for s in target["sentences"]]))
    print("")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duc2001", type=str, required=True)
    parser.add_argument(
        "--duc2002-documents", type=str, required=True)
    parser.add_argument(
        "--duc2002-summaries", type=str, required=True)
    parser.add_argument(
        "--output-dir", type=str, required=True)
    parser.add_argument(
        "--seed", type=int, default=43929524)
    parser.add_argument("--procs", type=int, required=False, default=None)
    args = parser.parse_args()
    random.seed(args.seed)

    if args.procs is None:
        args.procs = min(cpu_count(), 16)

    duc2001_dir = os.path.join(args.output_dir, "duc2001")
    duc2002_dir = os.path.join(args.output_dir, "duc2002")

    duc2001.preprocess_sds(duc2001_dir, nist_data_path=args.duc2001)
    duc2002.preprocess_sds(
        duc2002_dir,
        nist_document_data_path=args.duc2002_documents,
        nist_summary_data_path=args.duc2002_summaries)
    make_test_data(duc2002_dir, args.output_dir, args.procs)
    make_train_valid_data(duc2001_dir, args.output_dir, args.procs)

if __name__ == "__main__":
    main()
