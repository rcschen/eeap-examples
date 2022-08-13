'''
fix missing ng-vocab.tsv issue
https://github.com/sujitpal/eeap-examples/issues/3
'''
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import re
import os, codecs

DATA_DIR = "../data"
IDLABELS_FILE = os.path.join(DATA_DIR, "docsim-idlabels.tsv")
TEXTS_FILE = os.path.join(DATA_DIR, "docsim-texts.tsv")
NG_VOCAB = os.path.join(DATA_DIR, "ng-vocab.tsv")

VOCAB_SIZE = 40000
LABELS = {"similar": 1, "different": 0}

ng_data = fetch_20newsgroups(subset="all", 
                             data_home=DATA_DIR,
                             shuffle=True,
                             random_state=42)
num_docs = len(ng_data.data)
print("#-docs in dataset:", num_docs)

cvec = CountVectorizer(max_features=VOCAB_SIZE)

ids = np.arange(num_docs)
X = cvec.fit_transform(ng_data.data)
y = np.array(ng_data.target)
print("after vectorization:", X.shape, y.shape)
with codecs.open(NG_VOCAB, 'w', encoding='utf-8') as fw:
    for word, count in cvec.vocabulary_.items():
        try:
          print('>>>>>>',word, ' | ', count)
          fw.write('{}\t{}\n'.format(word, count))
        except Exception as e:
          print(e)


