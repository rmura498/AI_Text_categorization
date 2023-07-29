import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def removing_stop_words(dataframe):
    stop_words = set(stopwords.words('english'))
    for i in dataframe.index:

        new_title = ''
        new_body = ''

        title_text = dataframe["TITLE"][i]
        title_words = title_text.split()
        body_text = dataframe["BODY"][i]
        body_words = body_text.split()

        for t in title_words:
            if not t in stop_words:
                new_title = new_title + ' ' + t

        for b in body_words:
            if not b in stop_words:
                new_body = new_body + ' ' + b

        dataframe["TITLE"][i] = new_title
        dataframe["BODY"][i] = new_body

    return dataframe


def stemming(dataframe):
    ps = PorterStemmer()
    for i in dataframe.index:

        new_title = ''
        new_body = ''

        title_text = dataframe["TITLE"][i]
        title_words = word_tokenize(title_text)

        body_text = dataframe["BODY"][i]
        body_words = word_tokenize(body_text)
        for t in title_words:
            new_title = new_title + ' ' + ps.stem(t)
        for b in body_words:
            new_body = new_body + ' ' + ps.stem(b)

        dataframe["TITLE"][i] = new_title
        dataframe["BODY"][i] = new_body

    return dataframe
