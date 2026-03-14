
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns


def plot_accuracy(history, model_name):

    plt.figure()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title(model_name + " Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend(['Train','Validation'])

    plt.savefig("results/graphs/" + model_name + "_accuracy.png")
    plt.close()


def plot_loss(history, model_name):

    plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title(model_name + " Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(['Train','Validation'])

    plt.savefig("results/graphs/" + model_name + "_loss.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(model_name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("results/confusion_matrices/" + model_name + "_cm.png")
    plt.close()


def plot_roc(y_true, y_prob, model_name):

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
    plt.plot([0,1],[0,1],'--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(model_name + " ROC Curve")

    plt.legend(loc="lower right")

    plt.savefig("results/graphs/" + model_name + "_roc.png")
    plt.close()


def plot_precision_recall(y_true, y_prob, model_name):

    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure()

    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(model_name + " Precision-Recall Curve")

    plt.savefig("results/graphs/" + model_name + "_precision_recall.png")
    plt.close()
