import json
import pandas as pd
import numpy as np
import fasttext
import nltk

def load_review_dat(path, genre):
    
    # keep 10% of reviews
    reviews = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if np.random.uniform() < 0.1:
                reviews.append(entry)
                
    # convert to DF, retaining only relevant columns
    reviews = pd.DataFrame(reviews, columns=['user_id','book_id','rating','review_text','n_votes','n_comments'])
    reviews["n_likes"] = reviews["n_votes"] + reviews["n_comments"]
    reviews = reviews.drop(["n_votes","n_comments"],axis=1)
    reviews["genre"] = genre
    
    return(reviews)

# stack data
mystery = load_review_dat("data/goodreads_reviews_mystery.json", "mystery")
history = load_review_dat("data/goodreads_reviews_history_biography.json", "history")
fantasy = load_review_dat("data/goodreads_reviews_fantasy_paranormal.json", "fantasy")

reviews = pd.concat([mystery, history, fantasy], ignore_index=True)
print("Reviews Loaded")
print(reviews.groupby("genre").count()["user_id"])

# filter to reviews in English
language_model = fasttext.load_model('lid.176.bin')
pred = language_model.predict(reviews["review_text"].str.replace('\n','').to_list())
keep_ind = [i for i in range(len(pred[0])) if pred[0][i][0] == '__label__en' and pred[1][i][0] > .90]
reviews = reviews.iloc[keep_ind,].reset_index(drop=True)
print("Reviews Filtered to English")
print(reviews.groupby("genre").count()["user_id"])

# tokenize reviews
reviews["tokens"] = reviews["review_text"].apply(lambda review: [word.lower() for word in nltk.tokenize.word_tokenize(review) if word.isalpha()])

# remove stop words
stopwords = nltk.corpus.stopwords.words('english')
reviews["tokens"] = reviews["tokens"].apply(lambda review: [word for word in review if word not in stopwords])
print("Stop Words Removed")

# lemmatization
wnl = nltk.stem.wordnet.WordNetLemmatizer()
reviews["tokens"] = reviews["tokens"].apply(lambda review: [wnl.lemmatize(word) for word in review])
print("Lemmatized")

# save
reviews.to_pickle("data/reviews.pkl")
print("Saved")
