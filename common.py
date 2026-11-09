"""Shared helper functions used by part1.py ... part4.py.
Importing this file does NOT run any Part (there is no code at the bottom)."""
import csv
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "records.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")


# ---------------------------------------------------------------- files
def load_data():
    """Read data/records.csv. Returns X (800 x 7), outcome y, value column."""
    with open(DATA_FILE) as f:
        reader = csv.reader(f)
        next(reader)  # skip the header line
        rows = [r for r in reader]
    X = np.array([[float(v) for v in r[1:8]] for r in rows])  # X1 ... X7
    y = np.array([int(r[8]) for r in rows])                   # loan_default
    value = np.array([float(r[9]) for r in rows])             # monthly spend
    return X, y, value


def format_value(v):
    """Floats -> 4 decimals, ints -> ints, True/False -> yes/no, text stays text."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "%.4f" % v
    return str(v)


def write_results(filename, pairs):
    """Write 'key = value' lines. pairs is a list of (key, value)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, filename), "w") as f:
        for key, v in pairs:
            f.write("%s = %s\n" % (key, format_value(v)))


def write_csv(filename, header, rows):
    """Write a CSV file with the given header list and list of rows."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, filename), "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([format_value(v) for v in row])


# ---------------------------------------------------------------- Task 2.1
def make_split(y):
    """Stratified 70/30 split. Returns (train_index, test_index), both in file order."""
    train_index = []
    test_index = []
    for label in (0, 1):
        rows_of_label = [i for i in range(len(y)) if y[i] == label]
        cut = 7 * len(rows_of_label) // 10      # integer arithmetic
        train_index += rows_of_label[:cut]
        test_index += rows_of_label[cut:]
    return np.array(sorted(train_index)), np.array(sorted(test_index))


def standardise(train_X, test_X):
    """Use the TRAINING mean and sample SD (divide by n-1) for both sets."""
    mean = train_X.mean(axis=0)
    sd = train_X.std(axis=0, ddof=1)
    return (train_X - mean) / sd, (test_X - mean) / sd


def get_split_data():
    """Everything Part 2 and Part 4 need about the split, in one dictionary."""
    X, y, value = load_data()
    train_index, test_index = make_split(y)
    train_raw = X[train_index]
    test_raw = X[test_index]
    train_X, test_X = standardise(train_raw, test_raw)
    return {
        "train_raw": train_raw, "test_raw": test_raw,
        "train_X": train_X, "test_X": test_X,
        "train_y": y[train_index], "test_y": y[test_index],
    }


# ---------------------------------------------------------------- logistic regression
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(X, y, weights, bias):
    """Binary cross-entropy averaged over rows: -(1/n) sum[y ln p + (1-y) ln(1-p)].
    Written as mean(ln(1+e^z) - y*z), which is the same number but never gives log(0)."""
    z = X @ weights + bias
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def train_logistic(X, y, learning_rate, updates, record_at=()):
    """Batch gradient descent starting from zeros.
    Returns weights, bias and a dictionary {number_of_updates: training loss}."""
    n_rows, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    losses = {}
    if 0 in record_at:
        losses[0] = bce_loss(X, y, weights, bias)
    for step in range(1, updates + 1):
        p = sigmoid(X @ weights + bias)
        error = p - y                                  # (y_hat - y)
        grad_w = X.T @ error / n_rows                  # (1/n) sum (y_hat - y) x_j
        grad_b = error.sum() / n_rows                  # (1/n) sum (y_hat - y)
        weights = weights - learning_rate * grad_w
        bias = bias - learning_rate * grad_b
        if step in record_at:
            losses[step] = bce_loss(X, y, weights, bias)
    return weights, bias, losses


def predict_proba(X, weights, bias):
    return sigmoid(X @ weights + bias)


# ---------------------------------------------------------------- metrics
def confusion(y_true, predicted):
    """Returns tp, fp, tn, fn as plain ints."""
    tp = int(np.sum((predicted == 1) & (y_true == 1)))
    fp = int(np.sum((predicted == 1) & (y_true == 0)))
    tn = int(np.sum((predicted == 0) & (y_true == 0)))
    fn = int(np.sum((predicted == 0) & (y_true == 1)))
    return tp, fp, tn, fn


def safe_divide(a, b):
    """a / b, or 0 when the denominator is 0."""
    return a / b if b != 0 else 0.0


def metrics_from_cells(tp, fp, tn, fn):
    """Accuracy, precision, recall, F1 and specificity from confusion cells."""
    accuracy = safe_divide(tp + tn, tp + fp + tn + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    specificity = safe_divide(tn, tn + fp)
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "specificity": specificity}


# ---------------------------------------------------------------- KNN
def knn_positive_counts(query_X, ref_X, ref_y, max_k):
    """For every query row, count positive neighbours among the first K neighbours,
    for every K = 1 ... max_k. Returns an array (n_query x max_k): entry [i, K-1].
    Neighbours are ranked by (distance, position in file order) using a stable sort."""
    diff = query_X[:, None, :] - ref_X[None, :, :]
    distance = np.sqrt((diff ** 2).sum(axis=2))        # Euclidean distance
    order = np.argsort(distance, axis=1, kind="stable")
    labels_in_order = ref_y[order]                     # labels, nearest neighbour first
    return np.cumsum(labels_in_order[:, :max_k], axis=1)


def knn_predict_from_counts(counts, k):
    """Predicted class and probability for one K (K is odd)."""
    positives = counts[:, k - 1]
    predicted = (positives >= (k + 1) / 2).astype(int)
    probability = positives / k
    return predicted, probability


def knn_cross_validation(train_X, train_y, k_values):
    """5-fold CV on the training rows. Fold of a row = its position modulo 5.
    Returns {K: mean F1 over the five folds}."""
    n = len(train_y)
    fold_of_row = np.arange(n) % 5
    max_k = max(k_values)
    f1_lists = {k: [] for k in k_values}
    for fold in range(5):
        held_out = fold_of_row == fold
        counts = knn_positive_counts(train_X[held_out], train_X[~held_out],
                                     train_y[~held_out], max_k)
        for k in k_values:
            predicted, _ = knn_predict_from_counts(counts, k)
            tp, fp, tn, fn = confusion(train_y[held_out], predicted)
            f1_lists[k].append(metrics_from_cells(tp, fp, tn, fn)["f1"])
    return {k: sum(f1_lists[k]) / 5 for k in k_values}


def choose_best_k(cv_scores):
    """Highest mean F1; ties go to the smaller K."""
    best_k = None
    for k in sorted(cv_scores):
        if best_k is None or cv_scores[k] > cv_scores[best_k]:
            best_k = k
    return best_k
