"""Part 4 - Model comparison and architecture calculations.
Run from the project folder:  python part4.py
Writes outputs/part4_results.txt, part4_splits.csv, part4_voting.csv,
part4_adaboost.csv, part4_architecture.csv"""
import math

import numpy as np

from common import (get_split_data, train_logistic, predict_proba, confusion,
                    metrics_from_cells, knn_positive_counts, knn_predict_from_counts,
                    knn_cross_validation, choose_best_k, write_results, write_csv)


# ------------------------------------------------ Task 4.1  entropy and Gini
def entropy(positives, n):
    """-sum p*log2(p) over the two classes (0 for a pure or empty node)."""
    if n == 0:
        return 0.0
    total = 0.0
    for count in (positives, n - positives):
        p = count / n
        if p > 0:
            total -= p * math.log2(p)
    return total


def gini(positives, n):
    """1 - sum p^2 over the two classes."""
    if n == 0:
        return 0.0
    p = positives / n
    return 1.0 - p ** 2 - (1.0 - p) ** 2


def evaluate_split(condition, y):
    """condition is a True/False array (True = 'yes' side). Returns the node counts and gains."""
    n = len(y)
    n_yes = int(condition.sum())
    pos_yes = int(y[condition].sum())
    n_no = n - n_yes
    pos_no = int(y[~condition].sum())
    pos_all = int(y.sum())
    info_gain = entropy(pos_all, n) - (n_yes / n * entropy(pos_yes, n_yes)
                                       + n_no / n * entropy(pos_no, n_no))
    gini_gain = gini(pos_all, n) - (n_yes / n * gini(pos_yes, n_yes)
                                    + n_no / n * gini(pos_no, n_no))
    return n_yes, pos_yes, n_no, pos_no, info_gain, gini_gain


def make_conditions(raw):
    """The four candidate splits on ORIGINAL units (X1 = col 0, X3 = col 2, X6 = col 5, X7 = col 6)."""
    return {"S1": raw[:, 5] == 1, "S2": raw[:, 6] == 1,
            "S3": raw[:, 2] > 0.5, "S4": raw[:, 0] <= 24}


# ------------------------------------------------ Task 4.2  votes
def vote_row(name, y_true, predicted):
    tp, fp, tn, fn = confusion(y_true, predicted)
    m = metrics_from_cells(tp, fp, tn, fn)
    return [name, tp, fp, tn, fn, m["precision"], m["recall"], m["f1"]]


# ------------------------------------------------ Task 4.3  AdaBoost by hand
def adaboost_two_rounds():
    y = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])
    weights = np.full(8, 1.0 / 8)
    rows = []
    alphas = []
    errors = []
    for round_no, h in ((1, h1), (2, h2)):
        error = float(weights[h != y].sum())                  # weighted error
        alpha = 0.5 * math.log((1 - error) / error)
        unnormalised = weights * np.exp(-alpha * y * h)       # w * exp(-alpha * y * h)
        after = unnormalised / unnormalised.sum()
        for point in range(8):
            rows.append([round_no, point + 1, float(weights[point]),
                         float(unnormalised[point]), float(after[point])])
        errors.append(error)
        alphas.append(alpha)
        weights = after
    ensemble = np.sign(alphas[0] * h1 + alphas[1] * h2)
    accuracy = float(np.mean(ensemble == y))
    return errors, alphas, rows, accuracy


# ------------------------------------------------ Task 4.5  parameter counting
def conv_output_size(size, filter_size, padding, stride):
    return (size - filter_size + 2 * padding) // stride + 1


def main():
    d = get_split_data()
    train_X, test_X = d["train_X"], d["test_X"]
    train_y, test_y = d["train_y"], d["test_y"]
    out = []

    # ---- 4.1
    n_train = len(train_y)
    out.append(("root_entropy", entropy(int(train_y.sum()), n_train)))
    out.append(("root_gini", gini(int(train_y.sum()), n_train)))
    conditions = make_conditions(d["train_raw"])
    split_rows = []
    for name in ("S1", "S2", "S3", "S4"):
        split_rows.append([name] + list(evaluate_split(conditions[name], train_y)))
    write_csv("part4_splits.csv",
              ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"],
              split_rows)
    names = ["S1", "S2", "S3", "S4"]
    best_info = names[int(np.argmax([r[5] for r in split_rows]))]    # first max wins ties
    best_gini = names[int(np.argmax([r[6] for r in split_rows]))]
    out += [("best_split_by_info_gain", best_info), ("best_split_by_gini_gain", best_gini)]

    # ---- 4.2 stump, logistic, knn, hard and soft vote on the TEST rows
    best_row = split_rows[names.index(best_info)]
    stump_rate_yes = best_row[2] / best_row[1]
    stump_rate_no = best_row[4] / best_row[3]
    out += [("stump_rate_yes", stump_rate_yes), ("stump_rate_no", stump_rate_no)]
    test_yes = make_conditions(d["test_raw"])[best_info]
    stump_prob = np.where(test_yes, stump_rate_yes, stump_rate_no)

    weights, bias, _ = train_logistic(train_X, train_y, 0.1, 1000)
    logistic_prob = predict_proba(test_X, weights, bias)

    cv_scores = knn_cross_validation(train_X, train_y, list(range(1, 22, 2)))
    best_k = choose_best_k(cv_scores)
    counts = knn_positive_counts(test_X, train_X, train_y, best_k)
    knn_pred, knn_prob = knn_predict_from_counts(counts, best_k)

    logistic_pred = (logistic_prob >= 0.3).astype(int)
    stump_pred = (stump_prob >= 0.3).astype(int)
    hard_pred = ((logistic_pred + knn_pred + stump_pred) >= 2).astype(int)
    mean_prob = (logistic_prob + knn_prob + stump_prob) / 3
    soft_pred = (mean_prob >= 0.3).astype(int)
    write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
              [vote_row("logistic", test_y, logistic_pred), vote_row("knn", test_y, knn_pred),
               vote_row("stump", test_y, stump_pred), vote_row("hard_vote", test_y, hard_pred),
               vote_row("soft_vote", test_y, soft_pred)])

    # ---- 4.3
    errors, alphas, ada_rows, ensemble_accuracy = adaboost_two_rounds()
    out += [("round1_error", errors[0]), ("round1_alpha", alphas[0]),
            ("round2_error", errors[1]), ("round2_alpha", alphas[1])]
    write_csv("part4_adaboost.csv",
              ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], ada_rows)
    out.append(("ensemble_accuracy", ensemble_accuracy))

    # ---- 4.4
    out.append(("oob_fraction", (1 - 1 / 560) ** 560))

    # ---- 4.5 Model A option 1: MLP 784 -> 16 -> 16 -> 10 (weights + one bias per neuron)
    mlp_hidden1 = 784 * 16 + 16
    mlp_hidden2 = 16 * 16 + 16
    mlp_output = 16 * 10 + 10
    # Model A option 2: CNN
    size = 28
    cnn_conv1 = 5 * 5 * 1 * 6 + 6                       # 5x5 filters, 1 input channel, 6 filters
    size = conv_output_size(size, 5, 2, 1)              # 28
    size = conv_output_size(size, 2, 0, 2)              # pool -> 14
    cnn_conv2 = 5 * 5 * 6 * 16 + 16                     # 5x5 filters, 6 input channels, 16 filters
    size = conv_output_size(size, 5, 0, 1)              # 10
    size = conv_output_size(size, 2, 0, 2)              # pool -> 5
    flat = size * size * 16                             # 5 * 5 * 16 = 400
    cnn_dense1 = flat * 32 + 32
    cnn_dense2 = 32 * 10 + 10
    arch_rows = [["mlp_hidden1", mlp_hidden1], ["mlp_hidden2", mlp_hidden2],
                 ["mlp_output", mlp_output], ["cnn_conv1", cnn_conv1], ["cnn_conv2", cnn_conv2],
                 ["cnn_dense1", cnn_dense1], ["cnn_dense2", cnn_dense2]]
    write_csv("part4_architecture.csv", ["layer", "parameters"], arch_rows)
    out += [("mlp_total", mlp_hidden1 + mlp_hidden2 + mlp_output),
            ("cnn_total", cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2)]
    out.append(("fc_layer_params", 128 * 128 * 3 * 1024 + 1024))
    out.append(("conv_layer_params", 3 * 3 * 3 * 64 + 64))
    # Model B: RNN with 64-dim input, 128 hidden, 16 classes
    w_xh = 128 * 64
    w_hh = 128 * 128
    w_hy = 16 * 128
    out.append(("rnn_total", w_xh + w_hh + w_hy + 128 + 16))

    write_results("part4_results.txt", out)
    print("Part 4 done -> outputs/part4_*")


if __name__ == "__main__":
    main()
