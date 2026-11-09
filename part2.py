"""Part 2 - Loan default classification (train 560 rows / test 240 rows).
Run from the project folder:  python part2.py
Writes outputs/part2_results.txt, part2_thresholds.csv, part2_knn_cv.csv"""
import math

import numpy as np

from common import (get_split_data, train_logistic, predict_proba, bce_loss, confusion,
                    metrics_from_cells, knn_positive_counts, knn_predict_from_counts,
                    knn_cross_validation, choose_best_k, write_results, write_csv)

FALSE_ALARM_COST = 50     # INR per reminder sent to a borrower who would have paid
MISSED_COST = 400         # INR per default that was not flagged


def cost_of(y_true, probability, tau):
    """cost = 50 * fp + 400 * fn  when we flag every record with probability >= tau."""
    predicted = (probability >= tau).astype(int)
    tp, fp, tn, fn = confusion(y_true, predicted)
    return FALSE_ALARM_COST * fp + MISSED_COST * fn


def main():
    d = get_split_data()
    train_X, test_X = d["train_X"], d["test_X"]
    train_y, test_y = d["train_y"], d["test_y"]
    out = []

    # ---- Task 2.1
    out += [("train_rows", len(train_y)), ("train_positives", int(train_y.sum())),
            ("test_rows", len(test_y)), ("test_positives", int(test_y.sum()))]

    # ---- Task 2.2  (eta = 0.1, 1000 updates)
    weights, bias, losses = train_logistic(train_X, train_y, 0.1, 1000,
                                           record_at=(0, 1, 10, 100, 1000))
    for epoch in (0, 1, 10, 100, 1000):
        out.append(("loss_epoch_%d" % epoch, losses[epoch]))
    for j in range(7):
        out.append(("weight_X%d" % (j + 1), float(weights[j])))
    out.append(("bias", bias))
    out.append(("test_bce", bce_loss(test_X, test_y, weights, bias)))

    # ---- Task 2.3  (three learning rates, 200 updates)
    for eta in (0.01, 0.1, 1.0):
        _, _, lo = train_logistic(train_X, train_y, eta, 200, record_at=(200,))
        out.append(("loss_epoch_200_eta_%s" % eta, lo[200]))

    # ---- Task 2.4  odds ratio = e^weight
    for j in range(7):
        out.append(("odds_ratio_X%d" % (j + 1), math.exp(weights[j])))

    # ---- Task 2.5  two thresholds on the test rows
    test_prob = predict_proba(test_X, weights, bias)
    for tau in (0.5, 0.3):
        predicted = (test_prob >= tau).astype(int)
        tp, fp, tn, fn = confusion(test_y, predicted)
        m = metrics_from_cells(tp, fp, tn, fn)
        name = "tau_%s_" % tau
        out += [(name + "tp", tp), (name + "fp", fp), (name + "tn", tn), (name + "fn", fn),
                (name + "accuracy", m["accuracy"]), (name + "precision", m["precision"]),
                (name + "recall", m["recall"]), (name + "f1", m["f1"]),
                (name + "specificity", m["specificity"])]
    out.append(("blind_accuracy", float(np.mean(test_y == 0))))   # always predict 0

    # ---- Task 2.6  threshold sweep -> CSV
    rows = []
    for k in range(1, 10):
        tau = k / 10
        predicted = (test_prob >= tau).astype(int)
        tp, fp, tn, fn = confusion(test_y, predicted)
        rows.append([tau, tp, fp, tn, fn, tp / (tp + fn), fp / (fp + tn)])
    write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

    # ---- Task 2.7  KNN with 5-fold CV
    k_values = list(range(1, 22, 2))
    cv_scores = knn_cross_validation(train_X, train_y, k_values)
    write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], [[k, cv_scores[k]] for k in k_values])
    best_k = choose_best_k(cv_scores)
    counts = knn_positive_counts(test_X, train_X, train_y, best_k)
    predicted, _ = knn_predict_from_counts(counts, best_k)
    tp, fp, tn, fn = confusion(test_y, predicted)
    m = metrics_from_cells(tp, fp, tn, fn)
    out += [("knn_best_k", best_k), ("knn_tp", tp), ("knn_fp", fp), ("knn_tn", tn),
            ("knn_fn", fn), ("knn_accuracy", m["accuracy"]), ("knn_precision", m["precision"]),
            ("knn_recall", m["recall"]), ("knn_f1", m["f1"])]

    # ---- Task 2.8  choose tau with money, using TRAINING rows only
    train_prob = predict_proba(train_X, weights, bias)
    best_tau = None
    best_cost = None
    for k in range(1, 20):
        tau = k / 20
        c = cost_of(train_y, train_prob, tau)
        if best_cost is None or c < best_cost:     # strict "<" keeps the smaller tau on ties
            best_tau, best_cost = tau, c
    out += [("cost_tau", best_tau), ("cost_train_at_tau", best_cost),
            ("cost_test_at_tau", cost_of(test_y, test_prob, best_tau)),
            ("cost_test_at_0.5", cost_of(test_y, test_prob, 0.5)),
            ("cost_test_flag_nobody", MISSED_COST * int(test_y.sum()))]

    write_results("part2_results.txt", out)
    print("Part 2 done -> outputs/part2_*")


if __name__ == "__main__":
    main()
