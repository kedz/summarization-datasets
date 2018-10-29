summarization-datasets
======================
Pre-processing and in some cases downloading of datasets for the paper "Content Selection in Deep 
Learning Models of Summarization."

To install run:
```sh
$ python setup.py install
```

If you haven't installed spacy before in your current environment you should also run:
```sh
python -m spacy download en
```

Also it might be good to set your number of OMP threads to a small number, e.g. `export OMP_NUM_THREADS=2`

# CNN/DailyMail Dataset
To run:
```sh
python summarization-datasets/preprocess_cnn_dailymail.py \
    --data-dir data/
```
This will create the CNN/DM data in a directory `data/cnn-dailymail`.
This dataset is quite large and will take a while to preprocess. Grab a coffee!

# DUC Dataset

To obtain this data, first sign the release forms/email NIST 
(details here: https://duc.nist.gov/data.html).  

You should obtain from NIST, two files for the 2001/2002 data and a username and password.
Assuming you have the NIST data in the folder called `raw_data`, you should have following:
```
raw_data/DUC2001_Summarization_Documents.tgz
raw_data/DUC2002_Summarization_Documents.tgz
```
You will also need to download additional data from nist which you can do using a script
that will be in your bin directory after installation:
```sh
$ duc2002-test-data.sh USERNAME PASSWORD raw_data
```
where USERNAME and PASSWORD should have been given to you by NIST to access their website data.
This should create a file `raw_data/DUC2002_test_data.tar.gz`

Now run the preprocessing scripts:

```sh
python summarization-datasets/preprocess_duc_sds.py \
    --duc2001 raw_data/DUC2001_Summarization_Documents.tgz \
    --duc2002-documents raw_data/DUC2002_Summarization_Documents.tgz \
    --duc2002-summaries raw_data/DUC2002_test_data.tar.gz 
    --data-dir data
```

This will create put preprocessed duc data in `data/duc-sds/`.
 
# Reddit Dataset

To run:
```sh
python summarization-datasets/preprocess_reddit.py \
    --data-dir data/
```
This will create the Reddit data in a directory `data/reddit`. 
 
# PubMed Dataset

To run:
```sh
python summarization-datasets/preprocess_pubmed.py \
    --data-dir data/
```
This will create the PubMed data in a directory `data/pubmed`.

