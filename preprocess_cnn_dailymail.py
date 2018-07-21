import requests
import argparse
import sys
import os
import shutil
import zipfile
import tempfile
import spacy
from spacy.tokens import Doc
import hashlib
import ujson as json
import re
from multiprocessing import Pool, cpu_count
import rouge_papier


# Modified Abigail See's preprocessing code.

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


NUM_EXPECTED_CNN_STORIES = 92579
NUM_EXPECTED_DM_STORIES = 219506
dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()

def get_url_hashes(url_list):
    return [hashhex(url.encode("utf8")) for url in url_list]



REMAP = {"-LRB-": "(", "-RRB-": ")", "-LCB-": "{", "-RCB-": "}",
         "-LSB-": "[", "-RSB-": "]", "``": '"', "''": '"'}

def fix_summary(lines):
    text = "\n".join(lines)
    return re.sub(
        r"-LRB-|-RRB-|-LCB-|-RCB-|-LSB-|-RSB-|``|''",
        lambda m: REMAP.get(m.group()), text)

def fix_article(lines, nlp):
    inputs = []

    for line in lines:
        line = re.sub(
            r"-LRB-|-RRB-|-LCB-|-RCB-|-LSB-|-RSB-|``|''",
            lambda m: REMAP.get(m.group()), line)
        doc = nlp(line)
        for sent in doc.sents:
            text = sent.text.strip()
            tokens = [w.text.strip().lower() for w in sent]
            tc = len([w for w in tokens if len(w) > 0])
            if tc == 0:
                continue
            pos = [w.pos_ for w in sent]
            ne = [w.ent_type_ for w in sent]
            word_count = len(tokens)
            inputs.append({
                "text": text,
                "tokens": tokens,
                "pos": pos,
                "ne": ne,
                "word_count": word_count})
    for i, input in enumerate(inputs, 1):
        input["sentence_id"] = i
    return inputs


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
           ("stories directory {} contains {} "
            "files but should contain {}").format(
               stories_dir, num_stories, num_expected))

def download_file_from_google_drive(id, expected_size, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            size = 0
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    size += len(chunk)
                sys.stdout.write(
                    "[{:10d} of {:10d}]\r".format(size, expected_size))
                sys.stdout.flush()
            print("")
            if size != expected_size:
                raise Exception(
                    "Download failed! "
                    "Actual file size differs from expected file size!")

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def unzip_file(path, dest):
    try:
        zip_file = zipfile.ZipFile(path)
        zip_file.extractall(dest)
    finally:
        zip_file.close()

def download_urls(dest_dir):    

    def save_url(response, dest):
        CHUNK_SIZE = 32768
        with open(dest, "wb") as fp:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    fp.write(chunk)
    
    root_url = "https://github.com/abisee/cnn-dailymail/raw/master/url_lists/"
    train_url = root_url + "all_train.txt"
    val_url = root_url + "all_val.txt"
    test_url = root_url + "all_test.txt"

    train_url_path = os.path.join(dest_dir, "all_train.txt")
    val_url_path = os.path.join(dest_dir, "all_val.txt")
    test_url_path = os.path.join(dest_dir, "all_test.txt")

    session = requests.Session()
    save_url(session.get(train_url, stream=True), train_url_path)
    save_url(session.get(val_url, stream=True), val_url_path)
    save_url(session.get(test_url, stream=True), test_url_path)
    return train_url_path, val_url_path, test_url_path

def get_art_abs(story_file):
    lines = read_text_file(story_file)

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx,line in enumerate(lines):
        if line.strip() == "":
            continue # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

  # Make article into a single string
    article = article_lines
    abstract = highlights
    return article, abstract

def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line=="": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


def init_worker():
    global nlp
    nlp = spacy.load('en', parser=False)
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    

def preprocess_inputs(args):

    story_file, abstract_output_dir = args

    global nlp
    article, abstract = get_art_abs(story_file)
    abstract_text = fix_summary(abstract)
    inputs = fix_article(article, nlp)
    story_id = os.path.basename(story_file).split(".")[0]
    if len(inputs) == 0:
        print("\nBAD:", story_id)
        return None

    example = {"id": story_id, "inputs": inputs}

    abs_path = os.path.join(abstract_output_dir, story_id + ".spl")
    with open(abs_path, "w") as sfp:
        sfp.write(abstract_text)

    labels = get_labels(example, [abstract_text], 50)
    return json.dumps(example), json.dumps(labels)

def get_labels(example, summary_texts, sent_limit):
    input_texts = [input["text"] if input["word_count"] > 2 else "@@@@@@"
                   for input in example["inputs"]][:sent_limit]

    ranks, pairwise_ranks = rouge_papier.compute_extract(
        input_texts, summary_texts, mode="sequential", ngram=1,
        remove_stopwords=True, length=100)
    labels = [1 if r > 0 else 0 for r in ranks]
    if len(labels) < len(example["inputs"]):
        delta = len(example["inputs"]) - len(labels)
        labels.extend([0] * delta)

    return {"id": example["id"], "labels": labels}

def write_to_file(url_path, cnn_dir, dm_dir, story_output_path,
                  labels_output_path,
                  abstract_output_path, pool):

    url_list = read_text_file(url_path)
    url_hashes = get_url_hashes(url_list)
    story_fnames = [s+".story" for s in url_hashes]
    num_stories = len(story_fnames)
    
    story_paths = []
    for fn in story_fnames:
        if os.path.isfile(os.path.join(cnn_dir, fn)):
            story_paths.append(
                (os.path.join(cnn_dir, fn), abstract_output_path))
        elif os.path.isfile(os.path.join(dm_dir, fn)):
            story_paths.append(
                (os.path.join(dm_dir, fn), abstract_output_path))
        else:
            raise Exception("Missing file for story {}".format(fn))

    with open(story_output_path, "w") as story_fp, \
            open(labels_output_path, "w") as labels_fp:

        for idx, result in enumerate(
                pool.imap(preprocess_inputs, story_paths), 1):
            sys.stdout.write("Writing story {}/{}\r".format(idx, num_stories))
            sys.stdout.flush()

            if result is not None:
                ex_json, lbl_json = result
                story_fp.write(ex_json)
                story_fp.write("\n")
                labels_fp.write(lbl_json)
                labels_fp.write("\n")
 
    print("")

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def main():
    CNN_TOK_GID = "0BzQ6rtO2VN95cmNuc2xwUS1wdEE"
    CNN_TOK_EXPECTED_SIZE = 207268941
    DM_TOK_GID = "0BzQ6rtO2VN95bndCZDdpdXJDV1U"
    DM_TOK_EXPECTED_SIZE = 482735659

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--spacy-procs", type=int, required=False, default=None)
    args = parser.parse_args()

    if args.spacy_procs is None:
        args.spacy_procs = min(cpu_count(), 16)

    try:
        workdir = tempfile.mkdtemp()
        print(workdir)
        print(args.output_dir)
        print("Downloading train/val/test splits.")
        train_urls, val_urls, test_urls = download_urls(workdir)

        DM_TOK_ZIP = os.path.join(workdir, "dm_stories_tokenized.zip")
        CNN_TOK_ZIP = os.path.join(workdir, "cnn_stories_tokenized.zip")
        DM_TOK_STORIES = os.path.join(workdir, "dm_stories_tokenized")
        CNN_TOK_STORIES = os.path.join(workdir, "cnn_stories_tokenized")

        print("Downloading DailyMail data from googledrive.")
        download_file_from_google_drive(
            DM_TOK_GID, DM_TOK_EXPECTED_SIZE, DM_TOK_ZIP)
        print("Unpacking DailMail data.")
        unzip_file(DM_TOK_ZIP, workdir)

        print("Downloading CNN data from googledrive.")
        download_file_from_google_drive(
            CNN_TOK_GID, CNN_TOK_EXPECTED_SIZE, CNN_TOK_ZIP)
        print("Unpacking CNN data.")
        unzip_file(CNN_TOK_ZIP, workdir)

        check_num_stories(CNN_TOK_STORIES, NUM_EXPECTED_CNN_STORIES)    
        check_num_stories(DM_TOK_STORIES, NUM_EXPECTED_DM_STORIES)      

        train_stories = os.path.join(
            args.output_dir, "inputs", "cnn.dm.inputs.train.json")
        val_stories = os.path.join(
            args.output_dir, "inputs", "cnn.dm.inputs.valid.json")
        test_stories = os.path.join(
            args.output_dir, "inputs", "cnn.dm.inputs.test.json")

        train_labels = os.path.join(
            args.output_dir, "labels", "cnn.dm.labels.train.json")
        val_labels = os.path.join(
            args.output_dir, "labels", "cnn.dm.labels.valid.json")
        test_labels = os.path.join(
            args.output_dir, "labels", "cnn.dm.labels.test.json")

        train_abstracts = os.path.join(
            args.output_dir, "human-abstracts", "train")                                     
        valid_abstracts = os.path.join(
            args.output_dir, "human-abstracts", "valid")                                     
        test_abstracts = os.path.join(
            args.output_dir, "human-abstracts", "test")                                      
    
        check_dir(os.path.join(args.output_dir, "inputs"))
        check_dir(os.path.join(args.output_dir, "labels"))
        check_dir(train_abstracts)                                                 
        check_dir(valid_abstracts)                                                 
        check_dir(test_abstracts)
        
        pool = Pool(args.spacy_procs, initializer=init_worker)

        print("Writing cnn/dailymail validation data...")
        write_to_file(
            val_urls,
            CNN_TOK_STORIES,
            DM_TOK_STORIES,
            val_stories,
            val_labels,
            valid_abstracts,
            pool)

        print("Writing cnn/dailymail test data...")
        write_to_file(
            test_urls,
            CNN_TOK_STORIES,
            DM_TOK_STORIES,
            test_stories,
            test_labels,
            test_abstracts,
            pool)

    finally:
        shutil.rmtree(workdir)

if __name__ == "__main__":
    main()
