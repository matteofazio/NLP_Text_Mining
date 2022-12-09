# -*- coding: utf-8 -*-
"""
Utility functions for ML models.

@author: Andrea Belli <abelli@expert.ai>
"""
import time
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


def compute_prediction_latency(dataset, model, n_instances=-1):
    """Compute prediction latency of a model.
    
    The model must have a predict method.
    """
    if n_instances == -1:
        n_instances = len(dataset)
    start_time = time.process_time()
    model.predict(dataset)
    total_latency = time.process_time() - start_time
    return total_latency / n_instances


def from_encode_to_literal_labels(y_true, y_pred, idx2tag):
    '''Transform sequences of encoded labels in sequences of string labels'''
    let_y_true = list()
    let_y_pred = list()
    for sent_idx in range(len(y_true)):
        let_sent_true = []
        let_sent_pred = []
        for token_idx in range(len(y_true[sent_idx])):
            let_sent_true.append(idx2tag[y_true[sent_idx][token_idx]])
            let_sent_pred.append(idx2tag[y_pred[sent_idx][token_idx]])
        let_y_true.append(let_sent_true)
        let_y_pred.append(let_sent_pred)
    
    return let_y_true, let_y_pred
