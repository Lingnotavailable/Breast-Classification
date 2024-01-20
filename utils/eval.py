from __future__ import print_function, absolute_import
import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix, roc_curve, roc_auc_score, accuracy_score

__all__ = ['accuracy','classification_report']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def classification_report(output, target):
    
    output = output.clone().detach().cpu().numpy()
    target = target.clone().detach().cpu().numpy()

    pred = np.where(output < 0.5, 0, 1)

    acc = 100*accuracy_score(target, pred)
    precision = 100* precision_score(target, pred, zero_division=0, average='micro')
    recall = 100* recall_score(target, pred, zero_division=0,average='micro')
    f1 = f1_score(target, pred,zero_division=0,average='micro')
    cm = confusion_matrix(target, pred)


    try:
        auc_score = roc_auc_score(target,output)
    except ValueError:
        auc_score = 0
    try:
        fpr, tpr, thresholding = roc_curve(target,output)
    except ValueError:
        fpr = 0
        tpr = 0
    
    roc = {'fpr':fpr, 'tpr':tpr, 'auc': auc_score}
    metrics = {'acc':acc, 'prec': precision, 'recall':recall, 'f1': f1, 'auc_s':auc_score}

    return (metrics, roc, cm)