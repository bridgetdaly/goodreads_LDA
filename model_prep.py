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
dat = pd.read_csv("data/reviews.csv")
dat = dat.sample(n=102000, random_state=271)

# tokenize
dat["tokenized"] = dat["tokenized_words"].astype(str).apply(lambda review: [word for word in nltk.tokenize.word_tokenize(review)])

# split into train/test
train, test = train_test_split(dat,
                               test_size=2000,
                               random_state=271)

print(len(train), len(test))
print(train.groupby("genre").count()["user_id"]/len(train))

# collect train tokens and add common bigrams
reviews = train["tokenized"].tolist()
add_bigrams(reviews, min_count=20)

# build and filter vocabulary
dictionary = Dictionary(reviews)
dictionary.filter_extremes(no_below=20, no_above=0.35)
dictionary.save("data/dictionary.pkl", pickle_protocol=5)

# build and filter corpus (and reviews for use in coherence)
corpus = [dictionary.doc2bow(rev) for rev in reviews]
train_corpus = []
train_reviews = []
for r in range(len(reviews)):
    if len(corpus[r]) > 10:
        train_corpus.append(corpus[r])
        train_reviews.append(reviews[r])
MmCorpus.serialize("data/train_corpus.mm", train_corpus)
temp_file = open("data/train_reviews.pkl", "wb")
pickle.dump(train_reviews, temp_file)
temp_file.close()
print("Train corpus: ", len(train_corpus))

# build test corpus
test_reviews = test["tokenized"].tolist()
add_bigrams(test_reviews, min_count=1)
test_corpus = [dictionary.doc2bow(rev) for rev in test_reviews]
test_corpus = [review for review in test_corpus if len(review) > 10]
MmCorpus.serialize("data/test_corpus.mm", test_corpus)
print("Test corpus: ", len(test_corpus))