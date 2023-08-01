from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from operator import itemgetter

number_of_features = 800


def retrieving_most_discriminant_words(X_train, y_train):
    cv = CountVectorizer()
    X_vec = cv.fit_transform(X_train)

    IG = dict(zip(cv.get_feature_names_out(),
                  mutual_info_classif(X_vec, y_train, discrete_features=True, n_neighbors=3)))
    most_discriminant = dict(sorted(IG.items(), key=itemgetter(1), reverse=True)[:number_of_features])
    print('Number of words to consider:', len(most_discriminant))
    # print("The most discriminant words are " + str(most_discriminant))
    most_discriminant_words = list(most_discriminant.keys())

    return most_discriminant_words


def computing_tfidf(most_discriminant_words, X):

    print("shape ", X.shape)
    cv = TfidfVectorizer(max_features=number_of_features, vocabulary=most_discriminant_words)
    X_vec = cv.fit_transform(X)

    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_vec)
    idf = dict({'feature_name': cv.get_feature_names_out(), 'idf_weights': tfidf_transformer.idf_})
    tf_idf = pd.DataFrame(X_tfidf.toarray(), columns=cv.get_feature_names_out())
    X = X_tfidf.toarray()
    print("shape tfidf", X.shape)

    return X
