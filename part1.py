"""Part 1 - Estimation, inference and a regression baseline (all 800 rows).
Run from the project folder:  python part1.py
Writes outputs/part1_results.txt"""
import math

import numpy as np

from common import load_data, write_results


# ------------------------------------------------ small helper functions
def calculate_mean(values):
    return float(np.sum(values) / len(values))


def calculate_sample_sd(values):
    """Sample SD: square root of  sum((x - mean)^2) / (n - 1)."""
    mean = calculate_mean(values)
    return math.sqrt(float(np.sum((values - mean) ** 2)) / (len(values) - 1))


def closed_form_line(x, y):
    """Simple regression:  slope = cov(x, y) / var(x),  intercept = mean(y) - slope * mean(x)."""
    n = len(x)
    mean_x = calculate_mean(x)
    mean_y = calculate_mean(y)
    cov_xy = float(np.sum((x - mean_x) * (y - mean_y))) / (n - 1)
    var_x = float(np.sum((x - mean_x) ** 2)) / (n - 1)
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def split_residual_sd(fitted, residuals):
    """Sort rows by fitted value (stable, so ties keep row order). Return the sample SD
    of residuals in the lower 400 rows, the upper 400 rows, and upper / lower."""
    order = np.argsort(fitted, kind="stable")
    lower = residuals[order[:400]]
    upper = residuals[order[400:]]
    sd_lower = calculate_sample_sd(lower)
    sd_upper = calculate_sample_sd(upper)
    return sd_lower, sd_upper, sd_upper / sd_lower


def adjusted_r2(r2, n, p):
    """1 - (1 - R2) * (n - 1) / (n - p - 1)"""
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


# ------------------------------------------------ tasks
def task_1_1(value, out):
    n = len(value)
    mean_value = calculate_mean(value)
    sd_value = calculate_sample_sd(value)
    se_value = sd_value / math.sqrt(n)               # standard error = SD / sqrt(n)
    out += [("n", n), ("mean_value", mean_value), ("sd_value", sd_value),
            ("se_value", se_value),
            ("ci95_low", mean_value - 1.96 * se_value),
            ("ci95_high", mean_value + 1.96 * se_value)]
    return n, mean_value, sd_value, se_value


def task_1_2(mean_value, se_value, out):
    z_stat = (mean_value - 8.0) / se_value           # how many SEs away from 8.0
    out += [("z_stat", z_stat), ("reject_h0", abs(z_stat) > 1.96)]


def task_1_3(sd_value, out):
    margin = 0.5
    needed = (1.96 * sd_value / margin) ** 2         # n >= (1.96 * SD / E)^2
    out.append(("n_required", math.ceil(needed)))


def task_1_4(x4, value, out):
    n = len(value)
    slope, intercept = closed_form_line(x4, value)
    fitted = intercept + slope * x4
    mean_y = calculate_mean(value)
    tss = float(np.sum((value - mean_y) ** 2))       # total variation
    ess = float(np.sum((fitted - mean_y) ** 2))      # explained by the line
    rss = float(np.sum((value - fitted) ** 2))       # left over
    out += [("reg_slope", slope), ("reg_intercept", intercept),
            ("tss", tss), ("ess", ess), ("rss", rss),
            ("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss),
            ("r2", 1.0 - rss / tss),
            ("rmse", math.sqrt(rss / n)),
            ("mae", float(np.sum(np.abs(value - fitted))) / n)]
    return slope, intercept, fitted


def task_1_5(x4, value, reg_slope, out):
    n = len(value)
    mean_x = calculate_mean(x4)
    sd_x = calculate_sample_sd(x4)
    z = (x4 - mean_x) / sd_x
    m = 0.0
    c = 0.0
    eta = 0.1

    def cost(m, c):
        e = value - (m * z + c)
        return float(np.sum(e ** 2)) / n             # J = (1/n) sum e^2

    saved = {0: cost(m, c)}
    for step in range(1, 501):
        e = value - (m * z + c)
        grad_m = -(2.0 / n) * float(np.sum(z * e))   # dJ/dm
        grad_c = -(2.0 / n) * float(np.sum(e))       # dJ/dc
        m = m - eta * grad_m
        c = c - eta * grad_c
        if step in (1, 10, 100, 500):
            saved[step] = cost(m, c)
    gd_slope = m / sd_x                              # back to the original X4 scale
    gd_intercept = c - m * mean_x / sd_x
    out += [("gd_J0", saved[0]), ("gd_J1", saved[1]), ("gd_J10", saved[10]),
            ("gd_J100", saved[100]), ("gd_J500", saved[500]),
            ("gd_slope", gd_slope), ("gd_intercept", gd_intercept),
            ("gd_slope_gap", abs(gd_slope - reg_slope))]


def task_1_6(x4, value, fitted_raw, out):
    n = len(value)
    residual_raw = value - fitted_raw
    sd_low, sd_up, ratio = split_residual_sd(fitted_raw, residual_raw)
    out += [("resid_sd_lower_raw", sd_low), ("resid_sd_upper_raw", sd_up),
            ("resid_sd_ratio_raw", ratio)]

    ln_x = np.log(x4)
    ln_y = np.log(value)
    slope, intercept = closed_form_line(ln_x, ln_y)
    fitted = intercept + slope * ln_x
    mean_y = calculate_mean(ln_y)
    tss = float(np.sum((ln_y - mean_y) ** 2))
    rss = float(np.sum((ln_y - fitted) ** 2))
    r2 = 1.0 - rss / tss
    out += [("loglog_intercept", intercept), ("loglog_slope", slope),
            ("loglog_r2", r2), ("loglog_rmse", math.sqrt(rss / n))]
    sd_low, sd_up, ratio = split_residual_sd(fitted, ln_y - fitted)
    out += [("resid_sd_lower_log", sd_low), ("resid_sd_upper_log", sd_up),
            ("resid_sd_ratio_log", ratio)]
    return adjusted_r2(r2, n, 1)


def task_1_7(X, value, loglog_adj_r2, out):
    n = len(value)
    ln_y = np.log(value)
    # columns: constant, ln(X4), X1, X2, X3, X5, X6, X7
    design = np.column_stack([np.ones(n), np.log(X[:, 3]), X[:, 0], X[:, 1],
                              X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
    # normal equation: beta = (X'X)^-1 X'Y, solved as a linear system
    beta = np.linalg.solve(design.T @ design, design.T @ ln_y)
    fitted = design @ beta
    tss = float(np.sum((ln_y - calculate_mean(ln_y)) ** 2))
    rss = float(np.sum((ln_y - fitted) ** 2))
    r2 = 1.0 - rss / tss
    for i in range(8):
        out.append(("mr_beta_%d" % i, float(beta[i])))
    out += [("mr_r2", r2), ("mr_adj_r2", adjusted_r2(r2, n, 7)),
            ("loglog_adj_r2", loglog_adj_r2)]


def main():
    X, y, value = load_data()
    x4 = X[:, 3]
    out = []
    n, mean_value, sd_value, se_value = task_1_1(value, out)
    task_1_2(mean_value, se_value, out)
    task_1_3(sd_value, out)
    reg_slope, reg_intercept, fitted = task_1_4(x4, value, out)
    task_1_5(x4, value, reg_slope, out)
    loglog_adj = task_1_6(x4, value, fitted, out)
    task_1_7(X, value, loglog_adj, out)
    write_results("part1_results.txt", out)
    print("Part 1 done -> outputs/part1_results.txt")


if __name__ == "__main__":
    main()
