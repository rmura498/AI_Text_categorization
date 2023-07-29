import numpy as np
import pandas as pd
import pickle


def train_test_split():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    df = pd.read_pickle('dataframe.pkl')
    for i in df.index:

        if df['LEWISSPLIT'][i] == 'TRAIN':
            X_train.append(df['BODY'][i])
            y_train.append(df['TOPICS'][i])
        elif df['LEWISSPLIT'][i] == 'TEST':
            X_test.append(df['BODY'][i])
            y_test.append(df['TOPICS'][i])

    print('len x train {}, len y train {}, len x test {}, len y test {}'.format(len(X_train),
                                                                                len(y_train),
                                                                                len(X_test),
                                                                                len(y_test)))
    print(X_train[0])
    print(y_train[0])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    with open('X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)

    return X_train, X_test, y_train, y_test
