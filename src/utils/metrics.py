import numpy as np
from sklearn import metrics as skmetrics


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def tpr(labels, out):
    '''
    Returns the TPR
    '''
    ooh = np.zeros_like(out)
    ooh[np.arange(len(out)), np.argmax(out, axis=1)] = 1
    out = ooh
    loh = np.zeros_like(out)
    loh[np.arange(len(labels)), labels] = 1
    labels = loh

    out = (out > 0).astype(int)
    nout = out / (np.sum(out, axis=1, keepdims=True) + 1e-15)
    true_pos = (labels * nout).sum() / float(labels.shape[0])
    return true_pos


def auc(labels, out):
    '''
    Returns the average auc-roc score
    '''
    roc_auc = -1
    loh = np.zeros_like(out)
    loh[np.arange(len(out)), labels] = 1
    ooh = out

    # Remove classes without positive examples
    col_to_keep = (((ooh > 0).astype(int) + loh).sum(axis=0) > 0)
    loh = loh[:, col_to_keep]
    ooh = out[:, col_to_keep]

    fpr, tpr, _ = skmetrics.roc_curve(loh.ravel(), ooh.ravel())
    roc_auc = skmetrics.auc(fpr, tpr)

    return 2 * roc_auc - 1
