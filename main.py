import os.path
import tarfile
import pickle
import pandas as pd
from extracting_data import compile_data, compile_dictionary
from cleaning_dataset import removing_stop_words, stemming
from collections import Counter
from traint_test_split import train_test_split
from feature_selection import retrieving_most_discriminant_words, computing_tfidf
from classification import classify

number_of_classes = 5

if not os.path.exists('./dataset'):
    file = tarfile.open('reuters21578.tar.gz')
    file.extractall('./dataset')
    file.close()

if not os.path.exists('dataframe.pkl'):
    dataset = compile_data()
    dataset_dicts = [compile_dictionary(data) for data in dataset]
    dataset_dicts = [data for data in dataset_dicts if data['TOPICS'] != 'none']  # or data['REUTERS TOPICS'] == 'YES']

    # maintaining the rows of the most common label
    labels = [data["TOPICS"] for data in dataset_dicts]
    flat_labels = [item for sublist in labels for item in sublist]
    common_labels = Counter(flat_labels)
    most_common_labels = common_labels.most_common(number_of_classes)
    print(most_common_labels)
    most_common_labels = [most_common_labels[i][0] for i in range(number_of_classes)]

    print("Common labels:", common_labels)
    print('the most common labels are:', most_common_labels)

    with open('labels.pkl', 'wb') as f:
        pickle.dump(most_common_labels, f)
    dataset_dicts = [data for data in dataset_dicts if data['TOPICS'][0] in most_common_labels]

    for data in dataset_dicts:
        if data['BODY'] == '':
            data['BODY'] = data['TITLE']

    df = pd.DataFrame(dataset_dicts)
    df = removing_stop_words(df)
    df = stemming(df)
    df.to_pickle('dataframe.pkl')

with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

if not os.path.exists('most_discriminant_features.pkl'):
    X_train, X_test, y_train, y_test = train_test_split(labels)
    most_discriminant_features = retrieving_most_discriminant_words(X_train, y_train)

with open('most_discriminant_features.pkl', 'rb') as f:
    most_discriminant_features = pickle.load(f)
with open('X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
print('The most discriminant features are:', most_discriminant_features)

X_train = computing_tfidf(most_discriminant_features, X_train)
X_test = computing_tfidf(most_discriminant_features, X_test)

classify(X_train, y_train, X_test, y_test)


