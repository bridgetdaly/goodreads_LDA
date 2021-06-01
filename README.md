# Book Review LDA

Topic analysis on ~100,000 book reviews. Final project for Applied Bayesian Analysis (STATS271) Stanford Spring 2021. Dataset comes from the [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home), originally scraped from [Goodreads](https://www.goodreads.com/).

1. data_prep.py: text preprocessing including filtering to English reviews, tokenizing, removing stopwords, and lemmatizing
2. explore_reviews.ipynb: explore processed dataset and vocabulary for reviews
3. model_prep.py: add bigrams, build and filter vocabulary, build and filter train corpus and test corpus
4. lda.ipynb: run LDA on 10 different topic sizes
5. lda_evalutation.ipynb: evaluate topic distribution and key words for 20 topic model
6. hlda.ipynb: run hLDA with 3 levels in topic hierarchy
7. authortopic.ipynb: run author-topic model on 10 different topic sizes
8. author_evaluation.ipynb: evaluate toipc distribution and key words for 10 topic model
9. authorlesstopic.ipynb: run authorless-topic model on 10 different topic sizes
10. authorless_evaluation.ipynb: evaluate topic distribution and key words for 20 topic model and explore 50 topic model
11. hdp.ipynb: run HDP model (appendix)