import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from wilds.common.metrics.metric import Metric, ElementwiseMetric, MultiTaskMetric
from wilds.common.metrics.loss import ElementwiseLoss
from wilds.common.utils import avg_over_groups, minimum, maximum
import sklearn.metrics
from scipy.stats import pearsonr

def logits_to_score(logits):
    assert logits.dim() in (1,2)
    if logits.dim()==2: #multi-class logits
        assert logits.size(1)==2, "Only binary classification"
        score = F.softmax(logits, dim=1)[:,1]
    else:
        score = logits
    return score

def logits_to_pred(logits):
    assert logits.dim() in (1,2)
    if logits.dim()==2: #multi-class logits
        pred = torch.argmax(logits, 1)
    else:
        pred = (logits>0).long()
    return pred

def logits_to_binary_pred(logits):
    assert logits.dim() in (1,2)
    pred = (logits>0).long()
    return pred


class Accuracy(ElementwiseMetric):
    def __init__(self, prediction_fn=logits_to_pred, name=None):
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'acc'
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        return (y_pred==y_true).float()

    def worst(self, metrics):
        return minimum(metrics)

class MultiTaskAccuracy(MultiTaskMetric):
    def __init__(self, prediction_fn=logits_to_binary_pred, name=None):
        self.prediction_fn = prediction_fn  # should work on flattened inputs
        if name is None:
            name = 'acc'
        super().__init__(name=name)

    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        if self.prediction_fn is not None:
            flattened_y_pred = self.prediction_fn(flattened_y_pred)
        return (flattened_y_pred==flattened_y_true).float()

    def worst(self, metrics):
        return minimum(metrics)

class Recall(Metric):
    def __init__(self, prediction_fn=logits_to_pred, name=None, average='binary'):
        self.prediction_fn = prediction_fn
        if name is None:
            name = f'recall'
            if average is not None:
                name+=f'-{average}'
        self.average = average
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred, average=self.average, labels=torch.unique(y_true))
        return torch.tensor(recall)

    def worst(self, metrics):
        return minimum(metrics)

class F1(Metric):
    def __init__(self, prediction_fn=logits_to_pred, name=None, average='binary'):
        self.prediction_fn = prediction_fn
        if name is None:
            name = f'F1'
            if average is not None:
                name+=f'-{average}'
        self.average = average
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        score = sklearn.metrics.f1_score(y_true, y_pred, average=self.average, labels=torch.unique(y_true))
        return torch.tensor(score)

    def worst(self, metrics):
        return minimum(metrics)

class PearsonCorrelation(Metric):
    def __init__(self, name=None):
        if name is None:
            name = 'r'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        r = pearsonr(y_pred.squeeze().detach().cpu().numpy(), y_true.squeeze().detach().cpu().numpy())[0]
        return torch.tensor(r)

    def worst(self, metrics):
        return minimum(metrics)

def mse_loss(out, targets):
    assert out.size()==targets.size()
    if out.numel()==0:
        return torch.Tensor()
    else:
        assert out.dim()>1, 'MSE loss currently supports Tensors of dimensions > 1'
        losses = (out - targets)**2
        reduce_dims = tuple(list(range(1, len(targets.shape))))
        losses = torch.mean(losses, dim=reduce_dims)
        return losses

class MSE(ElementwiseLoss):
    def __init__(self, name=None):
        if name is None:
            name = 'mse'
        super().__init__(name=name, loss_fn=mse_loss)
