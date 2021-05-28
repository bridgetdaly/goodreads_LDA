import pandas as pd
import numpy as np
import pickle
from pprint import pprint
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary, MmCorpus
from hlda.sampler import HierarchicalLDA

# load/format data
dictionary = Dictionary.load("data/dictionary.pkl")
corpus = MmCorpus("data/train_corpus.mm")

vocab = list(dictionary.token2id.keys())
train_corpus = []
for review in corpus:
    train_corpus.append(np.concatenate([[tup[0]]*int(tup[1]) for tup in review]).tolist())
    
# https://github.com/joewandy/hlda

# using defaults alpha = 10, eta = 0.1, gamma = 1 (CRP smoothing)
n_samples = 50        # no of iterations for the sampler
display_topics = 5   # the number of iterations between printing
num_levels = 3        # the number of levels in the tree
n_words = 10          # most probable words to print for each topic

hlda = HierarchicalLDA(train_corpus, vocab, num_levels=num_levels)
hlda.estimate(n_samples, display_topics=display_topics, n_words=n_words)

temp_file = open("data/hlda.pkl", "wb")
pickle.dump(hlda, temp_file)
temp_file.close()