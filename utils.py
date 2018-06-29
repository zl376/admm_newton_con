import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


EPS = 1E-8


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def CE(weight, feature, label):
    """
    Cross entropy
    """
    p = sigmoid(np.dot(feature, weight))
    return - np.sum(label * np.log(p + EPS) + (1-label) * np.log(1-p + EPS), axis=0) / label.shape[0]


def g_CE(weight, feature, label):
    """
    Gradient of Cross entropy
    """
    p = sigmoid(np.dot(feature, weight))
    return np.dot(feature.T, p-label) / label.shape[0]


def h_CE(weight, feature, label):
    """
    Hessian of Cross entropy
    """
    p = sigmoid(np.dot(feature, weight))
    return np.dot(feature.T, np.dot(np.diag(p * (1-p)), feature)) / label.shape[0]


def examine(weight, feature, label):

    z = np.dot(feature, weight)
    p = sigmoid(z)

    # Gradient
    print('Gradient:\n|G_x*|={0:.2e}, |G_0|={1:.2e}'.format(norm(g_CE(weight, feature, label)), norm(g_CE(np.zeros_like(weight), feature, label))))
    
    # Confusion matrix, Precision, Recall
    label_pred = np.zeros(p.shape)
    label_pred[p >= 0.5] = 1
    tn, fp, fn, tp = confusion_matrix(label, label_pred).ravel()
    print('Confusion matrix:\n {}'.format(confusion_matrix(label, label_pred)))
    print('Precision {0:.3f}, Recall {1:.3f}'.format(tp/(tp+fp), tp/(tp+fn)))

    # 
    plt.plot(z, p, 'b*')
    plt.plot(z, label, 'ro')
    plt.title('True label vs. predicted score')
    plt.xlabel('w^T x')
    plt.show()
    
    # ROC curve
    [fpr, tpr, thresholds] = roc_curve(label, p)
    plt.plot(fpr, tpr, 'b-')
    plt.title('AUC = {:.3f}'.format(auc(fpr, tpr)))
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.show()
    
    return


def norm(x):
    return np.sqrt(np.dot(x.T, x))