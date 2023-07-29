from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from operator import itemgetter
import pickle


def retrieving_most_discriminant_words(X_train, y_train):
    cv = CountVectorizer(max_df=0.95, min_df=2,
                         max_features=10000)
    X_vec = cv.fit_transform(X_train)

    IG = dict(zip(cv.get_feature_names_out(),
                  mutual_info_classif(X_vec, y_train, discrete_features=True)))
    most_discriminant = dict(sorted(IG.items(), key=itemgetter(1), reverse=True)[:1000])

    print("The most discriminant words are " + str(most_discriminant))
    most_discriminant_words = list(most_discriminant.keys())
    print(most_discriminant_words)

    return most_discriminant_words

def computing_tfidf(most_discriminant_words, X):

    for i in range(len(X)):

        new_body = ''

        body_text = X[i]
        body_words = body_text.split()
        for b in body_words:
            if not b in most_discriminant_words:
                new_body = new_body + ' ' + b

        X[i] = new_body

    cv = CountVectorizer(max_features=1000)
    X_vec = cv.fit_transform(X)

    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_vec)
    idf = dict({'feature_name': cv.get_feature_names_out(), 'idf_weights': tfidf_transformer.idf_})
    # print(idf)
    print('idfshape', len(idf))
    tf_idf = pd.DataFrame(X_tfidf.toarray(), columns=cv.get_feature_names_out())
    # print(tf_idf)
    X = X_tfidf.toarray()
    print(X)
    print(X.shape)
    print(X[0].shape)
    print(X[0])

    return X