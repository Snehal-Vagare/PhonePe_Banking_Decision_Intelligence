Domain: Banking

# PhonePe Banking Decision Intelligence (capstone)

A small, beginner-friendly Python + NumPy project for the lending-risk team of a UPI payments platform. It estimates average monthly spend, predicts loan default, segments borrowers and sizes future neural networks. **All data is synthetic** (made by `generate_data.py`); nothing here is real PhonePe data or financial advice. Money is in Indian Rupees (INR).

Files: `generate_data.py` (provided script, unchanged) writes `data/records.csv`; `part1.py` to `part4.py` do the four Parts; `common.py` holds helper functions shared by Part 2 and Part 4; every result is written to `outputs/`.

## How to run

Python 3.10 or newer is needed. Run these commands, in this order, from the repository root:

```
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

`generate_data.py` must print `800 10 170 e85e7605b784f4142844f79ad2dd0b8e`. Each part script rewrites its files in `outputs/`; running them again gives identical files. (On Windows you can use `py` instead of `python`.)

## Part 1 findings

All numbers come from `outputs/part1_results.txt`.

- **Test of the budget claim (R1.1).** The average monthly spend is 8.9382 (INR 8,938) with a 95% interval of 8.3026 to 9.5738. The test of H0: mean = 8.0 gives `z_stat` = 2.8930, which is above the critical value 1.96, so **H0 is rejected**. A Type I error here would mean concluding that average spend is not INR 8,000 when it really is, and so rebuilding the budget for nothing. The gap is 0.9382 units, about INR 938 a month (about 12% above the budget assumption). That is statistically clear and also big enough to matter for a budget, though one number cannot say whether it matters for every product.
- **Sample size (R1.2).** To estimate the average within INR 500 at 95% confidence, `n_required` = 1293 records, but only 800 are available. With 800 rows the margin is about 0.64 (INR 636), so the team would need about 500 more records for the INR 500 target.
- **Gradient descent (R1.3).** `gd_slope_gap` = 0.0000 (below 0.001): the cost has a single bowl-shaped minimum, and 500 small steps of 0.1 walk all the way to its bottom, which is exactly where the closed-form line sits (the cost stops changing at 40.3698 by update 100).
- **Funnel check (R1.4).** The raw line (value on inflow, R² = 0.5196) has `resid_sd_ratio_raw` = 3.7525: residuals in the upper half of fitted values are about 3.8 times as spread out as in the lower half, so **yes, it shows a funnel**. After taking logs, `resid_sd_ratio_log` = 1.0396, so the log model **removes the funnel**.
- **Multiple regression (R1.5).** Predictors with an absolute coefficient above 0.05 are **ln(X4) inflow (1.1576), X2 failed transactions (0.0940), X3 credit utilisation (0.1067) and X6 irregular salary (0.1518)**. This is only a mechanical rule: units differ, and X1 (-0.0074 per month) is small per month but not per year. `mr_adj_r2` = 0.7403 is clearly higher than `loglog_adj_r2` = 0.6561, so the extra predictors do earn their place.

## Part 2 findings

All numbers come from `outputs/part2_results.txt`, `part2_thresholds.csv` and `part2_knn_cv.csv`. The split is 560 training rows (119 defaults) and 240 test rows (51 defaults).

- **Learning rates (R2.1).** After 200 updates the losses from lowest to highest are 0.3703 (eta 1.0), 0.3776 (eta 0.1) and 0.5091 (eta 0.01). A rate that is too small barely moves in 200 steps, while the larger rates reach the minimum; 1.0 was still stable here, but a rate that is too large can overshoot.
- **Weights and odds (R2.2).** The largest absolute weights are X3 credit utilisation (0.9013), X1 account age (-0.5554) and X2 failed transactions (0.5505). The largest odds ratio is 2.4629 for X3: a borrower one standard deviation higher on credit utilisation has about 2.5 times the odds of defaulting, other things equal.
- **Accuracy and thresholds (R2.3).** A model that predicts "no default" for everyone has `blind_accuracy` = 0.7875, only just below the 0.8042 accuracy at tau = 0.5, so accuracy alone says little when defaults are rare. Lowering tau from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 (27 missed defaults fell to 13) and lowered precision from 0.5455 to 0.4634.
- **Logistic vs KNN (R2.4).** At tau = 0.3 logistic has F1 0.5714 and recall 0.7451. KNN (best K = 5 by cross-validation) has F1 0.3696 and recall 0.3333, so logistic is clearly better on both.
- **What to deploy (R2.5).** I would deploy the **logistic model at tau = 0.15**. Costs on the test rows (INR 50 per false alarm, INR 400 per missed default) are INR 6,050 at tau = 0.15, INR 11,800 at tau = 0.5 and INR 20,400 if nobody is flagged. The threshold was chosen on the training rows because the test rows must stay unseen; choosing on them would make the test cost look better than it really is.

## Part 3 findings

All numbers come from `outputs/part3_results.txt`, `part3_segments.csv` and `part3_hierarchical.csv`.

- **Elbow (R3.1).** WCSS for K = 2 to 6 is 2262.6753, 1602.8471, 1494.7732, 1323.7157 and 1278.1691. The successive drops are 659.8282 (2 to 3), 108.0739 (3 to 4), 171.0575 (4 to 5) and 45.5466 (5 to 6). The big drop ends at K = 3 and the rule gives `elbow_k` = 3, so K = 3 is used. The drops do not shrink steadily (4 to 5 is bigger than 3 to 4) because the pinned starting rows give one reproducible solution, not necessarily the global optimum.
- **Segments (R3.2).**
  - **A, "Steady veterans"** (322 records): about 47.5 months of tenure, low utilisation 0.2250, few failed transactions (1.95), score 62.2, outcome rate 0.0186. Action: keep them happy with automatic limit increases and light-touch reminders.
  - **B, "Heavy active users"** (255 records): 28.6 months, the highest inflow (28.7) and spend (17.12), the most failed transactions (6.07), outcome rate 0.2353. Action: watch failed payments and offer autopay.
  - **C, "New stretched borrowers"** (223 records): 14.4 months, high utilisation 0.7118, low score 38.5, **highest outcome rate 0.4664**. Action: early reminder calls before the due date and smaller starting limits.
- **Hierarchical clustering (R3.3).** At the cut, single linkage gives 4 clusters: E1, E4, E5 and E2+E3+E6. Complete linkage gives 2 clusters: E1+E4+E5 and E2+E3+E6 (average linkage also gives 2). Single linkage joins clusters through their closest pair, so its last merge happens at only 3.4 and the largest gap sits early; complete linkage uses the farthest pair, so the two groups stay apart until 7.9 and the big gap is at the end.
- **PCA (R3.4).** The mean vector is all zeros after standardising. PC1 (52.27% of the variance) has positive loadings for tenure X1 (0.5317) and score X5 (0.4418) and negative loadings for utilisation X3 (-0.5187), failed transactions X2 (-0.4100) and inflow X4 (-0.2916): it is a "steady and reliable versus stretched" axis. Two components keep 75.93% and **3 components** are needed for 80% (86.16%). The mean PC1 scores are A = 1.7773, B = -0.7831 and C = -1.6709, so PC1 orders the segments from the safest (A) to the riskiest (C).

## Part 4 findings

All numbers come from `outputs/part4_results.txt` and the `part4_*.csv` files.

- **Splits (R4.1).** Root entropy is 0.7462 and root Gini 0.3347. The best split by both measures is **S4 (account age at most 24 months)**: information gain 0.1252 and Gini gain 0.0606. It sends 188 records with 86 defaults (45.7%) to one side and 372 records with 33 defaults (8.9%) to the other, giving the purest children. A tree grown until every leaf is pure would memorise noise in the training rows and overfit; a maximum depth or a minimum number of rows per leaf stops it growing too deep.
- **Model comparison (R4.2).** On the test rows (F1 / recall): logistic 0.5714 / 0.7451, KNN 0.3696 / 0.3333, stump 0.5532 / 0.7647, hard vote 0.5344 / 0.6863, soft vote 0.5507 / 0.7451. I would ship the **logistic model**. Combining models did not help: neither vote beat logistic on F1, and the hard vote lost recall.
- **AdaBoost (R4.3).** Round 1 error is 0.2500 (alpha 0.5493); round 2 error is 0.1667 (alpha 0.8047). After round 1 the two misclassified points (3 and 7) carry a total weight of 0.5000 (0.2500 each). Round 2 trusts h2 more because h2 has a lower weighted error on the reweighted points (0.1667 against 0.2500). The two-round ensemble gets 0.7500 of the eight points right.
- **Out-of-bag share.** `oob_fraction` = 0.3676: about 37% of rows are left out of each bootstrap sample.
- **Architecture sizes (R4.4).** Model A option 2 (CNN) for a 28 x 28 x 1 cheque-digit input:

| Layer | Output shape | Parameters |
|---|---|---|
| Input | 28 x 28 x 1 | 0 |
| Conv 5x5, 6 filters, pad 2 (+ ReLU) | 28 x 28 x 6 | 156 |
| Max-pool 2x2, stride 2 | 14 x 14 x 6 | 0 |
| Conv 5x5, 16 filters, pad 0 (+ ReLU) | 10 x 10 x 16 | 2416 |
| Max-pool 2x2, stride 2 | 5 x 5 x 16 | 0 |
| Flatten | 400 | 0 |
| Dense 32 (+ ReLU) | 32 | 12832 |
| Dense 10 | 10 | 330 |

  The CNN has `cnn_total` = 15734 parameters and the fully connected option 784-16-16-10 has `mlp_total` = 13002. A fully connected layer on a 128 x 128 x 3 input with 1,024 neurons needs 50,332,672 parameters, while a convolutional layer with 64 filters of 3 x 3 needs only 1,792, because the same small filters are reused over the whole image.
  - **Model B (RNN):** W_xh is 128 x 64, W_hh is 128 x 128 and W_hy is 16 x 128, plus biases b_h (128) and b_y (16), so `rnn_total` = 26768. It does not grow with longer sequences because the same weights are reused at every time step.
  - **Activations:** ReLU in hidden layers, and softmax at the output with categorical cross-entropy loss for the 10 classes.
  - **Deep networks:** in a very deep plain network the gradient is multiplied by many small numbers on its way back, so early layers receive almost no learning signal; a skip connection adds the layer's input to its output, giving the gradient a short direct path back.

## Model-choice memo

To: Head of Lending Risk. From: Analytics.

I recommend the **logistic regression model**, flagging a borrower when the predicted default probability is at least **0.15**. On the 240 held-out test records it flags with F1 0.5714 and recall 0.7451 at tau = 0.3; KNN reaches only F1 0.3696, and neither a hard vote nor a soft vote beat it.

The threshold was chosen on training rows using INR 50 per needless reminder and INR 400 per missed default. On the test records this costs **INR 6,050**, against INR 11,800 at tau = 0.5 and INR 20,400 if nobody is flagged. Credit utilisation is the strongest signal (odds ratio 2.4629 per standard deviation), and new accounts default far more often (45.7% for accounts of 24 months or less, against 8.9%).

Main risk: only 51 test defaults exist, from one split of synthetic data, so the cost figures are noisy and the INR 50 and INR 400 costs are assumptions. Re-check them on fresh data before relying on them.

Neither Model A (cheque digits) nor Model B (payment remarks) is worth building before this structured model is in production.

## Limitations

1. **Synthetic data.** Every row comes from a generator, so real borrowers, real default rates and real spend patterns may differ a lot. Nothing here is real PhonePe data.
2. **One split and few positives.** All test results use a single 560/240 split with only 51 test defaults. A few records changing could move the costs, the best K and the chosen threshold; they are valid for this split only.
3. **Assumed costs and a skewed spend column.** The INR 50 and INR 400 costs are given assumptions, and the spend column is strongly right-skewed (SD 9.1723 above its mean 8.9382), so a normal-based interval and z-test are only approximate even with 800 rows.
4. **Pinned choices.** K-means uses fixed starting rows and may not reach the best clustering for every K. KNN cross-validated F1 is noisy (for example K = 1 and K = 3 tie at 0.4249).
5. **Not tested on new data.** No time-based or out-of-sample check was done, and the neural-network sizes in Part 4 are only counted, not trained.
