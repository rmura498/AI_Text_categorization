import os.path
import tarfile
import pickle
import pandas as pd
from extracting_data import compile_data, compile_dictionary
from cleaning_dataset import removing_stop_words, stemming
from collections import Counter
from traint_test_split import train_test_split
from feature_selection import retrieving_most_discriminant_words, computing_tfidf

if not os.path.exists('./dataset'):
    file = tarfile.open('reuters21578.tar.gz')
    file.extractall('./dataset')
    file.close()

if not os.path.exists('dataframe.pkl'):
    dataset = compile_data()
    dataset_dicts = [compile_dictionary(data) for data in dataset]
    dataset_dicts = [data for data in dataset_dicts if data['TOPICS'] != 'none']  # or data['REUTERS TOPICS'] == 'YES']

    # maintaining the rows of the 5 most common label
    labels = [data["TOPICS"] for data in dataset_dicts]
    common_labels = Counter(labels)
    most_common_labels = common_labels.most_common(5)
    most_common_labels = [most_common_labels[i][0] for i in range(5)]
    print('the most common labels are:', most_common_labels)
    with open('labels.pkl', 'wb') as f:
        pickle.dump(most_common_labels, f)
    dataset_dicts = [data for data in dataset_dicts if data['TOPICS'] in most_common_labels]
    for data in dataset_dicts:
        if data['BODY'] == '':
            data['BODY'] = data['TITLE']


    df = pd.DataFrame(dataset_dicts)
    df = removing_stop_words(df)
    df = stemming(df)
    df.to_pickle('dataframe.pkl')

X_train, X_test, y_train, y_test = train_test_split()

most_discriminant_words = retrieving_most_discriminant_words(X_train, y_train)
X_train = computing_tfidf(most_discriminant_words, X_train)
X_test = computing_tfidf(most_discriminant_words, X_test)





# 0 eliminare stop word
# 0.1 eliminare termini senza capacità discriminante (articoli e preposizioni)
# 0.2 eseguire lemmantizzazione ridurre ogni parola alla sua redice (make, making, made -> mak)
# 0.3 individuare le 5 classi
# https://pypi.org/project/PorterStemmer/
# 1. feature selection
# 1.1 valutare la capacità discriminante di ciascun termine nel training set, selezionare gli N termini più discriminanti
# con N da definire, e rappresentare ogni documento con un vettore di occorrenze, frequenze o altro (per esempio TF-IDF sez 5.1 eq 1)
# degli N termini selezionati
# 1.2 per valutare la capacità discriminante si può usare una delle misure definite nella sezione 5.4 e.g inform gain.

# TODO


# 2. addestrare M classificatori one vs all (le 5 classi con più documenti)

# 3 valutazione performances con F_\beta combinazione di precision e recall
# 3.1 usare versione macro.averaged, sezione 7 pag 36
