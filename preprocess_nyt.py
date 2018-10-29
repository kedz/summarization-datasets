import rouge_papier

import argparse
import pathlib

import tarfile
from bs4 import BeautifulSoup
from multiprocessing import Pool
import spacy
import re
import ujson as json


def get_paths(root_dir):
    data_dir = root_dir / "data"
    paths = []
    years = [x for x in data_dir.glob("*")]
    years.sort()
    for year in years:
        months = [x for x in year.glob("*")]
        months.sort()
        for month in months:
            if month.name.endswith('tgz'):
                paths.append(month)
    return paths

bad_sections = set([
    "Style", "Home and Garden", "Paid Death Notices", "Automobiles",
    "Real Estate", "Week in Review", "Corrections", "The Public Editor",
    "Editors' Notes"])

def get_article_text(xml):
    return "\n\n".join([p.get_text() for p in xml.find_all("p")])

def init_worker():
    global nlp
    nlp = spacy.load('en', parser=False)

def prepare_example(article_text, abstract_text, ol_text, doc_id, sections):
    global nlp
    inputs = []
    for doc in nlp.pipe(article_text.split("\n")):
        for sent in doc.sents:
            tokens_all = [w for w in sent
                          if w.text.strip() != '']
            if len(tokens_all) == 0:
                continue
            tokens = [w.text.strip().lower() for w in tokens_all]
            pos = [w.pos_ for w in tokens_all]
            ne = [w.ent_type_ for w in tokens_all]
            pretty_text = sent.text.strip()
            pretty_text = re.sub(r"\r|\n|\t", r" ", pretty_text)
            pretty_text = re.sub(r"\s+", r" ", pretty_text)
            inputs.append({"tokens": tokens, "text": pretty_text,
                           "pos": pos, "ne": ne, 
                           "word_count": len(pretty_text.split())})
    for i, inp in enumerate(inputs, 1):
        inp["sentence_id"] = i

    summary_texts = []
    if len(abstract_text) > 0:
        summary_texts.append(abstract_text)
    if len(ol_text) > 0:
        summary_texts.append(ol_text)
    input_texts = [inp["text"] if inp["word_count"] > 2 else "@@@@@"
                   for inp in inputs[:50]]
    ranks, pairwise_ranks = rouge_papier.compute_extract(
        input_texts, summary_texts, mode="sequential", ngram=1, 
        remove_stopwords=True, length=100)
    labels = [1 if r > 0 else 0 for r in ranks]
    if len(labels) < len(inputs):
        labels.extend([0] * (len(inputs) - len(labels)))
    labels = {"id": doc_id, "labels": labels}
    example = {"id": doc_id, "inputs": inputs, "sections": sections}
    return example, labels, abstract_text, ol_text

def extract_doc(content):
    soup = BeautifulSoup(content, "lxml")
   
    sections = set() 
    for meta in soup.find_all("meta"):
        if meta["name"] == "online_sections":
            for section in meta["content"].split(";"):
                section = section.strip()
                sections.add(section)

    if len(sections.intersection(bad_sections)) > 0:
        return None

    article_xml = soup.find("block", {"class": "full_text"})
    if article_xml is None:
        return None

    article_text = get_article_text(article_xml)
    if len(article_text.split()) < 200:
        return None
      
    abstract_xml = soup.find("abstract")
    if abstract_xml is not None:
        abs_txt = abstract_xml.get_text()
    else:
        abs_txt = ""

    online_lead_xml = soup.find(
        "block", {"class": "online_lead_paragraph"})
    if online_lead_xml is not None:
        online_lead_txt = online_lead_xml.get_text()
    else: 
        online_lead_txt = ""
    if len(abs_txt.split()) + len(online_lead_txt.split()) < 100:
        return None
    doc_id = soup.find("doc-id")["id-string"] 
    return article_text, abs_txt, online_lead_txt, doc_id, sections

def worker(args):
    content, inputs_dir, labels_dir, abs_dir = args

    # Process xml to get document and summary text. 
    doc_data = extract_doc(content)
    if doc_data is None:
        return False
    article_text, abs_txt, online_lead_txt, doc_id, sections = doc_data

    example, labels, abstract_text, ol_text = prepare_example(
        article_text, abs_txt, online_lead_txt, doc_id, sections)

    assert abstract_text == abs_txt
    assert online_lead_txt == ol_text

    inputs_path = inputs_dir / "{}.json".format(example["id"])
    inputs_path.write_text(json.dumps(example))
    labels_path = labels_dir / "{}.json".format(example["id"])
    labels_path.write_text(json.dumps(labels))

    if len(abs_txt) > 0:
        abs_path1 = abs_dir / "{}.1.txt".format(example["id"])
        abs_path1.write_text(abs_txt)
    if len(ol_text) > 0:
        abs_path2 = abs_dir / "{}.2.txt".format(example["id"])
        abs_path2.write_text(ol_text)

    return True

def doc_iter(tar_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            f = tar.extractfile(member)
            if f is None:
                continue
            content = f.read().decode("utf8")
            yield content

def preprocess_part(tar_paths, inputs_dir, labels_dir, abs_dir, procs=16):

    inputs_dir.mkdir(exist_ok=True, parents=True)
    labels_dir.mkdir(exist_ok=True, parents=True)
    abs_dir.mkdir(exist_ok=True, parents=True)

    def data_iter():
        for tar_path in tar_paths:
            for content in doc_iter(tar_path):
                yield content, inputs_dir, labels_dir, abs_dir
    pool = Pool(procs, initializer=init_worker)
    count = 0
    for i, is_good in enumerate(pool.imap(worker, data_iter()), 1):
        if is_good:
            count += 1
            print("{}".format(count), end="\r", flush=True)
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nyt", type=pathlib.Path, required=True)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--procs", type=int, required=False, default=None)
    args = parser.parse_args()

    if args.procs is None:
        args.procs = min(cpu_count(), 16)

    paths = get_paths(args.nyt)

    train_paths = paths[:-30]
    valid_paths = paths[-30:-18]
    test_paths = paths[-18:]
    print(train_paths[0], train_paths[-1])
    print(valid_paths[0], valid_paths[-1])
    print(test_paths[0], test_paths[-1])

    preprocess_part(
        valid_paths, 
        args.data_dir / "nyt" / "inputs" / "valid",
        args.data_dir / "nyt" / "labels" / "valid",
        args.data_dir / "nyt" / "human-abstracts" / "valid",
        procs=args.procs)

    preprocess_part(
        test_paths, 
        args.data_dir / "nyt" / "inputs" / "test",
        args.data_dir / "nyt" / "labels" / "test",
        args.data_dir / "nyt" / "human-abstracts" / "test",
        procs=args.procs)

    preprocess_part(
        train_paths, 
        args.data_dir / "nyt" / "inputs" / "train",
        args.data_dir / "nyt" / "labels" / "train",
        args.data_dir / "nyt" / "human-abstracts" / "train",
        procs=args.procs)

if __name__ == "__main__":
    main()
