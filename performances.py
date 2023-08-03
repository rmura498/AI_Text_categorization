from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import fbeta_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def performances(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0.0)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall", recall)
    print("F_beta score:", fbeta_score(y_test, y_pred, average='macro', beta=1))

