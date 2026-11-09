"""Part 3 - Segmentation and dimensionality reduction (all 800 rows, X1 to X5).
Run from the project folder:  python part3.py
Writes outputs/part3_results.txt, part3_segments.csv, part3_hierarchical.csv"""
import numpy as np

from common import load_data, write_results, write_csv


# ------------------------------------------------ Task 3.1  K-means
def standardise_columns(data):
    """(value - column mean) / column sample SD."""
    return (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)


def run_kmeans(data, k):
    """K-means with the pinned start rows 0, 800//k, 2*(800//k), ...
    Returns labels, centroids, number of assignment passes."""
    n = len(data)
    step = n // k
    centroids = data[[i * step for i in range(k)]].copy()
    labels = None
    passes = 0
    while True:
        passes += 1
        # squared distance from every row to every centroid
        dist = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist, axis=1)       # ties -> lower-numbered centroid
        if labels is not None and np.array_equal(new_labels, labels):
            break                                  # a pass where nobody moved
        labels = new_labels
        for c in range(k):
            members = data[labels == c]
            if len(members) > 0:                   # empty cluster keeps its centroid
                centroids[c] = members.mean(axis=0)
    return labels, centroids, passes


def calculate_wcss(data, labels, centroids):
    """Sum over all rows of the squared distance to the row's own centroid."""
    return float(((data - centroids[labels]) ** 2).sum())


# ------------------------------------------------ Task 3.4  hierarchical clustering
NAMES = ["E1", "E2", "E3", "E4", "E5", "E6"]
DISSIMILARITY = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def linkage_distance(cluster_a, cluster_b, rule):
    """Distance between two clusters (lists of member numbers) for single/complete/average."""
    pair_distances = [DISSIMILARITY[i][j] for i in cluster_a for j in cluster_b]
    if rule == "single":
        return min(pair_distances)
    if rule == "complete":
        return max(pair_distances)
    return sum(pair_distances) / len(pair_distances)       # average


def cluster_name(members):
    return "+".join(NAMES[i] for i in sorted(members))


def run_hierarchical(rule):
    """Returns a list of 5 merges: (name_a, name_b, height)."""
    clusters = [[i] for i in range(6)]
    merges = []
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = linkage_distance(clusters[a], clusters[b], rule)
                if best is None or d < best[0]:
                    best = (d, a, b)
        d, a, b = best
        first, second = clusters[a], clusters[b]
        if min(second) < min(first):               # cluster_a has the smaller first member
            first, second = second, first
        merges.append((cluster_name(first), cluster_name(second), d))
        merged = sorted(clusters[a] + clusters[b])
        clusters = [c for i, c in enumerate(clusters) if i not in (a, b)] + [merged]
    return merges


def clusters_at_cut(merges):
    """The cut is in the largest gap between consecutive merge heights."""
    heights = [m[2] for m in merges]
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    merges_below_cut = gaps.index(max(gaps)) + 1
    return 6 - merges_below_cut


# ------------------------------------------------ Task 3.5  PCA
def run_pca(z):
    """The seven PCA steps on the standardised matrix z (800 x 5)."""
    n = len(z)
    mean_vector = z.mean(axis=0)                   # step 1 (all ~0 after standardising)
    centred = z - mean_vector                      # step 2
    cov = centred.T @ centred / (n - 1)            # step 3
    values, vectors = np.linalg.eigh(cov)          # step 4
    order = np.argsort(values)[::-1]               # step 5: largest first
    values = values[order]
    vectors = vectors[:, order]
    for j in range(vectors.shape[1]):              # sign rule: largest |entry| positive
        if vectors[np.argmax(np.abs(vectors[:, j])), j] < 0:
            vectors[:, j] = -vectors[:, j]
    top_two = vectors[:, :2]                       # step 6
    projection = centred @ top_two                 # step 7
    return values, vectors, projection


def main():
    X, y, value = load_data()
    raw = X[:, :5]                                 # X1 ... X5 in original units
    z = standardise_columns(raw)
    out = []

    # ---- 3.1 K-means for K = 2..6
    wcss = {}
    for k in range(2, 7):
        labels, centroids, passes = run_kmeans(z, k)
        wcss[k] = calculate_wcss(z, labels, centroids)
        out.append(("wcss_k%d" % k, wcss[k]))
        if k == 3:
            labels3 = labels
        print("K=%d: %d assignment passes" % (k, passes))

    # ---- 3.2 elbow rule
    elbow_k = None
    for k in (3, 4, 5):
        if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k]):
            elbow_k = k
            break
    out.append(("elbow_k", elbow_k if elbow_k is not None else "none"))

    # ---- 3.3 profile the K = 3 segments (A, B, C by ascending mean X3)
    cluster_ids = sorted(range(3), key=lambda c: raw[labels3 == c, 2].mean())
    segment_names = ["A", "B", "C"]
    segment_of_row = np.zeros(len(raw), dtype=int)   # 0,1,2 = A,B,C
    seg_rows = []
    for s, c in enumerate(cluster_ids):
        mask = labels3 == c
        segment_of_row[mask] = s
        seg_rows.append([segment_names[s], int(mask.sum())] +
                        [float(raw[mask, j].mean()) for j in range(5)] +
                        [float(y[mask].mean()), float(value[mask].mean())])
        out.append(("size_segment_%s" % segment_names[s], int(mask.sum())))
    write_csv("part3_segments.csv",
              ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5",
               "outcome_rate", "mean_value"], seg_rows)

    # ---- 3.4 hierarchical clustering
    hier_rows = []
    for rule in ("single", "complete", "average"):
        merges = run_hierarchical(rule)
        for number, (a, b, height) in enumerate(merges, start=1):
            hier_rows.append([rule, number, a, b, height])
        out.append(("%s_clusters_at_cut" % rule, clusters_at_cut(merges)))
    write_csv("part3_hierarchical.csv",
              ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hier_rows)

    # ---- 3.5 PCA
    values, vectors, projection = run_pca(z)
    total = float(values.sum())
    out.append(("eigenvalue_sum", total))
    for i in range(5):
        out.append(("eigenvalue_%d" % (i + 1), float(values[i])))
    for i in range(5):
        out.append(("explained_ratio_%d" % (i + 1), float(values[i]) / total))
    cumulative = np.cumsum(values) / total
    for i in range(5):
        out.append(("cumulative_ratio_%d" % (i + 1), float(cumulative[i])))
    out.append(("components_for_80pct", int(np.argmax(cumulative >= 0.8)) + 1))
    for pc in (1, 2):
        for j in range(5):
            out.append(("pc%d_loading_X%d" % (pc, j + 1), float(vectors[j, pc - 1])))
    for r in range(3):
        out.append(("row%d_pc1" % (r + 1), float(projection[r, 0])))
        out.append(("row%d_pc2" % (r + 1), float(projection[r, 1])))
    for s in range(3):
        out.append(("mean_pc1_segment_%s" % segment_names[s],
                    float(projection[segment_of_row == s, 0].mean())))

    write_results("part3_results.txt", out)
    print("Part 3 done -> outputs/part3_*")


if __name__ == "__main__":
    main()
