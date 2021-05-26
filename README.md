# Book Review LDA

Topic analysis on ~100,000 book reviews. Final project for Applied Bayesian Analysis (STATS271) Stanford Spring 2021. Dataset comes from the [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home), originally scraped from [Goodreads](https://www.goodreads.com/).

1. data_prep.py: text preprocessing including filtering to English reviews, tokenizing, removing stopwords, and lemmatizing
2. explore_reviews.ipynb: explore processed dataset and vocabulary for reviews
3. model_prep.py: add bigrams, build and filter vocabulary, build and filter train corpus and test corpus
4. lda.ipynb: run LDA on 5 different topic sizes and evaluate coherence (errors when running as .py so had to dump all logging/output into the notebook)
