from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import fbeta_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(cm_normalised, annot=True, linewidths=0, square=False,
                     cmap="Greens", vmin=0, vmax=np.max(cm_normalised),
                     fmt=".2f", annot_kws={"size": 20})
    ax.set(xlabel='Predicted label', ylabel='True label')
    plt.show()


def performances(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0.0)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall", recall)
    print("F_beta score:", fbeta_score(y_test, y_pred, average='macro', beta=1))
    plot_confusion_matrix(y_test, y_pred)
