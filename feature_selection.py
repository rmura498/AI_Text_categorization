from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd
import pickle
import numpy as np
from operator import itemgetter

number_of_features = 800


def retrieving_most_discriminant_words(X_train, y_train):
    cv = CountVectorizer()
    X_vec = cv.fit_transform(X_train)

    # Compute mutual information for each label individually
    mi_per_label = []
    for label_idx in range(y_train.shape[1]):
        mi = mutual_info_classif(X_vec, y_train[:, label_idx], discrete_features=True, n_neighbors=3)
        mi_per_label.append(mi)

    # Calculate the average mutual information across all labels
    average_mi = np.mean(mi_per_label, axis=0)

    IG = dict(zip(cv.get_feature_names_out(), average_mi))
    most_discriminant = dict(sorted(IG.items(), key=itemgetter(1), reverse=True)[:number_of_features])
    print('Number of feature to consider:', len(most_discriminant))
    most_discriminant_features = list(most_discriminant.keys())
    with open('most_discriminant_features.pkl', 'wb') as f:
        pickle.dump(most_discriminant_features, f)
    return most_discriminant_features


def computing_tfidf(most_discriminant_features, X):
    print("shape ", X.shape)
    cv = TfidfVectorizer(max_features=number_of_features, vocabulary=most_discriminant_features)
    X_vec = cv.fit_transform(X)

    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_vec)
    idf = dict({'feature_name': cv.get_feature_names_out(), 'idf_weights': tfidf_transformer.idf_})
    tf_idf = pd.DataFrame(X_tfidf.toarray(), columns=cv.get_feature_names_out())
    X = X_tfidf.toarray()
    print("shape tfidf", X.shape)

    return X
