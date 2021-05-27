import pandas as pd
import numpy as np
import nltk
import pickle
from sklearn.model_selection import train_test_split
from gensim.models import Phrases
from gensim.corpora import Dictionary, MmCorpus
import logging
logging.basicConfig(filename="model_prep.log", format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# reference: https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html

# helper function to add bigram tokens
def add_bigrams(reviews, min_count):
    bigram = Phrases(reviews, min_count=min_count)
    for idx in range(len(reviews)):
        for token in bigram[reviews[idx]]:
            if "_" in token:
                # Token is a bigram, add to document.
                reviews[idx].append(token)

# load data
dat = pd.read_pickle("data/reviews.pkl")
dat = dat.sample(n=102000, random_state=271)

# split into train/test
train, test = train_test_split(dat,
                               test_size=2000,
                               random_state=271)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

print(len(train), len(test))
print(train.groupby("genre").count()["user_id"]/len(train))

# collect train tokens and add common bigrams
reviews = train["tokens"].tolist()
add_bigrams(reviews, min_count=20)

# build vocabulary
dictionary = Dictionary(reviews)
dictionary.filter_extremes(no_below=20, no_above=0.35)
dictionary.save("data/dictionary.pkl", pickle_protocol=5)

# build train corpus
corpus = [dictionary.doc2bow(rev) for rev in reviews]
train_corpus = []
for r in range(len(reviews)):
    if len(corpus[r]) > 10:
        train_corpus.append(corpus[r])
    else:
        train.drop(index=r, inplace=True)
MmCorpus.serialize("data/train_corpus.mm", train_corpus)
temp_file = open("data/train.pkl", "wb")
pickle.dump(train, temp_file)
temp_file.close()
print("Train corpus: ", len(train_corpus))

# build test corpus
test_reviews = test["tokens"].tolist()
add_bigrams(test_reviews, min_count=1)
test_corpus_pre = [dictionary.doc2bow(rev) for rev in test_reviews]
test_corpus = []
for r in range(len(test_reviews)):
    if len(test_corpus_pre[r]) > 10:
        test_corpus.append(test_corpus_pre[r])
    else:
        test.drop(index=r, inplace=True)
MmCorpus.serialize("data/test_corpus.mm", test_corpus)
temp_file = open("data/test.pkl", "wb")
pickle.dump(test, temp_file)
temp_file.close()
print("Test corpus: ", len(test_corpus))

# build authorless-topic model inputs
# 1) tsv with doc id, author, space separated tokens
genre_doc = train.iloc[:,5:]
genre_doc["tokens"] = genre_doc["tokens"].apply(lambda rev: " ".join(rev))
genre_doc.to_csv("data/genre_reviews.tsv", sep="\t", header=False)

sent_doc = train.iloc[:,[2,6]]
sent_doc["tokens"] = sent_doc["tokens"].apply(lambda rev: " ".join(rev))
sent_doc.to_csv("data/sent_reviews.tsv", sep="\t", header=False)

#2) vocabulary
vocab = pd.Series(dictionary.token2id.keys())
vocab.to_csv("data/vocab.tsv", index=False, header=False)