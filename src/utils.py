import torch
import numpy as np
import random
from sklearn.metrics import classification_report, roc_auc_score, f1_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_masks(data, train_ratio=0.7, val_ratio=0.1):
    num_nodes = data.num_nodes
    indices = np.random.permutation(num_nodes)

    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data


# FIXED class weights (handles imbalance better)
def compute_class_weights(y):
    class_counts = np.bincount(y.cpu().numpy())
    total = sum(class_counts)

    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float)


def evaluate_model(model, data, mask):
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

    y_true = data.y[mask].cpu()
    y_pred = pred[mask].cpu()

    print(classification_report(y_true, y_pred, zero_division=0))

    try:
        probs = torch.exp(out[mask])[:, 1].cpu()
        roc = roc_auc_score(y_true, probs)
        print("ROC-AUC:", roc)
    except:
        print("ROC-AUC could not be computed")

    f1 = f1_score(y_true, y_pred)
    print("F1 Score:", f1)


def get_risk_scores(model, data):
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.exp(out)

    return probs[:, 1]


def generate_alerts(scores, threshold=0.9):
    return scores > threshold