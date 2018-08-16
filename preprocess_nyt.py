import argparse
import os
import sys
import tarfile
from bs4 import BeautifulSoup
from multiprocessing import Pool
import spacy
import re
import rouge_papier
import ujson as json

def get_paths(root_dir):
    data_dir = os.path.join(root_dir, "data")
    paths = []
    years = os.listdir(data_dir)
    years.sort()
    for year in years:
        months = os.listdir(os.path.join(data_dir, year))
        months.sort()
        for month in months:
            path = os.path.join(data_dir, year, month)
            if path.endswith('tgz'):
                paths.append(path)
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
#    for i, inp in enumerate(inputs, 1):
#        print(i, " ".join(inp["tokens"]))
#    print("")

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

def worker(path): 
    
    data = []  
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            f = tar.extractfile(member)
            if f is None:
                continue
            content = f.read().decode("utf8")
            soup = BeautifulSoup(content, "lxml")
           
            sections = set() 
            for meta in soup.find_all("meta"):
                if meta["name"] == "online_sections":
                    for section in meta["content"].split(";"):
                        section = section.strip()
                        #if section not in gsections:
                        #    gsections.add(section)
                        #    print(section)
                        sections.add(section)

            if len(sections.intersection(bad_sections)) > 0:
                continue

            article_xml = soup.find("block", {"class": "full_text"})
            if article_xml is None:
                continue

            article_text = get_article_text(article_xml)
            if len(article_text.split()) < 200:
                continue
              
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
                continue 
            doc_id = soup.find("doc-id")["id-string"] 
            data.append(
                prepare_example(article_text, abs_txt, online_lead_txt, 
                                doc_id, sections))

    return data

def check_dir(dir_path):
    if dir_path != "" and not os.path.exists(dir_path):
        os.makedirs(dir_path)
 
def write_data(inputs_path, labels_path, summary_dir, paths, procs=16):

    pool = Pool(procs, initializer=init_worker)
    with open(inputs_path, "w") as inp_fp, open(labels_path, "w") as lbl_fp: 
        for i, results in enumerate(pool.imap(worker, paths), 1):
            sys.stdout.write("{}/{}\r".format(i, len(paths)))
            sys.stdout.flush()
            for example, labels, abs_text, ol_text in results:
                inp_fp.write(json.dumps(example))
                inp_fp.write("\n")
                lbl_fp.write(json.dumps(labels))
                lbl_fp.write("\n")
                if len(abs_text) > 0:
                    abs_path1 = os.path.join(
                        summary_dir, "{}.1.txt".format(example["id"]))
                    with open(abs_path1, "w") as fp:
                        fp.write(abs_text)
                if len(ol_text) > 0:
                    abs_path2 = os.path.join(
                        summary_dir, "{}.2.txt".format(example["id"]))
                    with open(abs_path2, "w") as fp:
                        fp.write(ol_text)
    print("")                   
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    paths = get_paths(args.input)

    train_paths = paths[:-30]
    valid_paths = paths[-30:-18]
    test_paths = paths[-18:]
    print(train_paths[0], train_paths[-1])
    print(valid_paths[0], valid_paths[-1])
    print(test_paths[0], test_paths[-1])

    inputs_dir = os.path.join(args.output, "inputs")
    check_dir(inputs_dir)
    labels_dir = os.path.join(args.output, "labels")
    check_dir(labels_dir)    

    valid_inputs = os.path.join(
        args.output, "inputs", "nyt.spacy.inputs.valid.json")
    valid_labels = os.path.join(
        args.output, "labels", "nyt.spacy.labels.valid.json")
    valid_summary_dir = os.path.join(
        args.output, "human_abstracts", "valid")
    check_dir(valid_summary_dir)
    write_data(valid_inputs, valid_labels, valid_summary_dir, valid_paths)

    test_inputs = os.path.join(
        args.output, "inputs", "nyt.spacy.inputs.test.json")
    test_labels = os.path.join(
        args.output, "labels", "nyt.spacy.labels.test.json")
    test_summary_dir = os.path.join(
        args.output, "human_abstracts", "test")
    check_dir(test_summary_dir)
    write_data(test_inputs, test_labels, test_summary_dir, test_paths)

    train_inputs = os.path.join(
        args.output, "inputs", "nyt.spacy.inputs.train.json")
    train_labels = os.path.join(
        args.output, "labels", "nyt.spacy.labels.train.json")
    train_summary_dir = os.path.join(
        args.output, "human_abstracts", "train")
    check_dir(train_summary_dir)
    write_data(train_inputs, train_labels, train_summary_dir, train_paths)





       #print(path, count)


               
                
#news?
#U.S.
#New York and Region
#World
#Business
#Technology
#Washington
#Health
#Arts
#Sports

#Automobiles
#Real Estate
#Week in Review

# REMOVE 
#Corrections
#The Public Editor
#Editors' Notes

#take both abstracts
#Opinion
#Education
#Theater
#Movies
#Books
#Science
#Dining and Wine
#Travel
#Magazine
#Job Market



#hard 
#Style
#Home and Garden

#bad 
##Paid Death Notices

if __name__ == "__main__":
    main()
