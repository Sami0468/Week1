# Adult Income Classification — Project README

Full end-to-end supervised ML project on the UCI Adult (Census Income) dataset, covering Week 1, Day 1 through Day 5: problem framing, EDA, baselines, preprocessing, feature engineering, model comparison, hyperparameter tuning, calibration, final validation, interpretation, and inference.

**Project status: functional, partially verified, not yet production-ready.** Several Day 4/5 items are still open — see *Known Limitations* below before relying on any "final" number.

---

## 1. Project Objective

Predict whether an individual's annual income exceeds $50,000/year from demographic and employment attributes, in order to support a targeted-outreach-style business decision (e.g. flag likely high earners for a marketing or screening workflow while avoiding wasted effort on people who don't qualify). The chosen business framing prioritizes **precision on the flagged (positive) group** over blindly maximizing recall, since the cost of acting on a false positive (wasted outreach) was defined as the more expensive error for this use case.

## 2. Dataset Description

- **Source:** UCI Adult / Census Income dataset, loaded via `sklearn.datasets.fetch_openml('adult', version=2, as_frame=True)` (mirrors the UCI repository, avoids raw-file quirks — see note below).
- **Size:** ~48,842 rows, mixed categorical and numeric demographic/employment fields (age, workclass, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country).
- **Class balance:** approximately **23.9% positive** (>50K) vs **76.1% negative** (≤50K) — a moderate class imbalance that shaped both the baseline choice and the primary-metric decision (see Section 6).
- **Known raw-file gotchas (avoided by using `fetch_openml`):** the original UCI files use `" ?"` (leading space) for missing values rather than plain `"?"`, and the raw `adult.test` file has a trailing period on target labels (`">50K."` vs `">50K"`), which breaks naive train/test concatenation if not handled.

## 3. Target Variable

- **Name:** `income`
- **Definition:** binary — `1` if annual income > $50,000, `0` otherwise (originally encoded as text labels `>50K` / `<=50K`, mapped to `1` / `0`).
- **Positive class:** `income > 50K` (the class the business objective is defined against).

## 4. Feature Engineering

Eight engineered features were added on top of the raw fields, built via a leak-safe `FunctionTransformer` that only ever reads the current row (no cross-row aggregation, no use of the target column — this was an explicit requirement to avoid information leakage into cross-validation folds):

| Feature | Type | Creation rule | Rationale |
|---|---|---|---|
| Age bucket | Categorical | Binned `age` into ranges | Income tends to rise then plateau/decline with age — a non-linear relationship a single numeric feature can't capture for linear models |
| Hours-per-week bin | Categorical | Binned `hours_per_week` | Part-time vs. full-time vs. overtime patterns relate to income differently than a raw linear hours count |
| Capital-gain flag | Boolean | `capital_gain > 0` | Most rows have zero capital gain; the binary presence/absence signal is stronger than the raw skewed value |
| Log capital-gain | Numeric | `log(capital_gain + 1)` | Capital gain is heavily right-skewed; log-scaling compresses extreme values for models sensitive to scale |
| Higher-education flag | Boolean | `education_num >= 13` (bachelor's or higher) | Directly mirrors the Day 1 rule-based baseline, which showed strong univariate signal |
| Education × hours interaction | Numeric | `education_num * hours_per_week` | Captures that the income effect of extra hours worked differs by education level |
| *(two additional engineered features)* | — | — | Logged with mutual-information scores in the Day 3 feature dictionary |

Each feature was scored with mutual information against the target and logged in a feature dictionary (name, type, creation rule, predictive signal) before being wired into the pipeline.

## 5. Preprocessing Steps

A single `ColumnTransformer`, fit once on the training split and reused unchanged for every model on every later day:

- **Numeric branch:** `SimpleImputer(strategy="median")` → `StandardScaler`
  *Median chosen over mean* because several numeric fields (notably `capital_gain`) are heavily right-skewed, so the median is far less distorted by outliers.
- **Categorical branch:** `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`
  *One-hot chosen over ordinal/label encoding* because none of the categorical fields (workclass, occupation, marital status, etc.) have a natural order; `handle_unknown="ignore"` protects inference against categories not seen during training.
- All `"?"` values were converted to `NaN` before imputation.
- Engineered features (Section 4) are appended via `FunctionTransformer` inside the same pipeline, so **no preprocessing step ever needs to be manually repeated outside the saved pipeline object.**

## 6. Models Tested

| Stage | Model | Test Accuracy | Test F1 | ROC AUC |
|---|---|---|---|---|
| Day 1 baseline | Majority-class predictor | ~0.761 (= negative rate) | 0 (positive class) | n/a |
| Day 1 baseline | Rule-based (`education_num≥13` OR `capital_gain>0`) | computed, see Task 4 CSV | computed, see Task 4 CSV | computed, see Task 4 CSV |
| Day 2 | Logistic Regression | 0.8527 | — | 0.9039 |
| Day 2 | Decision Tree | 0.8183 | — | 0.7522 |
| Day 3 (5-fold CV, engineered features) | Gradient Boosting | — | 0.712 | 0.928 |
| Day 4 (tuned, test set) | Logistic Regression | — | 0.674 | — |
| Day 4 (tuned, test set) | Random Forest | — | 0.691 | — |
| Day 4 (**not** tuned) | Gradient Boosting | — | 0.712 (carried from Day 3) | 0.928 (carried from Day 3) |

The primary evaluation metric used throughout was **F1 / ROC AUC** on the held-out test set (chosen at the end of Day 1 once a single fixed decision rule proved too blunt on its own); the classification threshold is a separate, still-open decision (see Section 9).

## 7. Hyperparameter Tuning Approach

- **Method:** `RandomizedSearchCV`
- **Scoring:** F1 (matches the project's primary metric)
- **Cross-validation:** `StratifiedKFold(k=5)`, identical folds used for both model comparison (Day 3) and tuning (Day 4)
- **Parallelism:** `n_jobs=-1`
- **Search budget:** realistic range (n_iter in the 50–100 range as specified by the task)
- **Search space per model:**
  - Logistic Regression: `penalty` (l1/l2), `C` (inverse regularization strength)
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
  - Gradient Boosting: `learning_rate`, `max_iter`/`n_estimators`, `max_depth`, `l2_regularization` — **specified but not yet run** (open item, see Section 9)

## 8. Best Parameters

Logistic Regression and Random Forest completed `RandomizedSearchCV`; their winning test F1 scores (0.674 and 0.691 respectively) are confirmed and verified. **The exact winning hyperparameter values are stored in the search object's `best_params_` output inside the Day 4 notebook / saved pipeline metadata** rather than restated here, to avoid this README drifting out of sync with the actual artifact — always treat the saved pipeline as the source of truth for exact parameter values.

Gradient Boosting has no best-parameters entry yet, since it was never put through the search (open item — see Section 9).

## 9. Selected Final Model

- **Best raw score in the project:** Gradient Boosting, F1 0.712 / ROC AUC 0.928 — but this is the **untuned**, Day 3 default configuration.
- **Best *tuned* model on record:** Random Forest, test F1 0.691.
- **Current recommendation:** treat Random Forest as the provisional final model until Gradient Boosting has actually been tuned. Once tuned, Gradient Boosting is very likely to become the final model, given it already leads untuned — but that claim is not yet verified and should not be assumed.

## 10. Classification Threshold

**Not yet finalized.** Probability calibration (Brier score, calibration curve, `CalibratedClassifierCV` if needed) and threshold selection were required Day 4 deliverables but have no executed, numeric output on record — no Brier score value, no chosen threshold, and no before/after confusion matrix currently exist for this project. Until this is completed, any inference code defaults to scikit-learn's implicit 0.5 threshold, which has **not** been validated as the correct business threshold for this task's precision-first objective. This is the single highest-priority open item in the project (see Section 9 of the full report / Known Limitations below).

## 11. Final Test Performance

See the table in Section 6. The most defensible current "final" numbers are Random Forest's test F1 of 0.691 (fully tuned and verified) or, provisionally, Gradient Boosting's F1 of 0.712 / ROC AUC of 0.928 (best raw score, but untuned — treat as an upper-bound estimate, not a finished number). A fully correct final-validation table (accuracy, precision, recall, F1, ROC AUC, PR AUC, and Brier score together, on the untouched hold-out test set) is a Day 5 deliverable that is still pending completion of the Day 4 calibration step.

## 12. Important Features

- From the Decision Tree (Day 2) and Random Forest / Gradient Boosting importance rankings: `capital_gain`, `education_num` (and the derived higher-education flag), `age` / age-bucket, and `hours_per_week` (and its interaction with education) consistently rank among the most influential fields.
- Logistic Regression coefficients (mapped back to one-hot feature names via `get_feature_names_out`) largely agree in direction with the tree-based importances — e.g. higher education, being married, and higher capital gain all push predictions toward the positive class, while very low weekly hours pushes toward the negative class.
- A full ranked list (top 10 positive / top 10 negative coefficients, plus tree feature importances) is produced in the Day 2 and Day 5 (Task 3) notebooks.

## 13. Known Limitations

- **Gradient Boosting, the best-performing model in the project, has never been hyperparameter-tuned.** This should be resolved before any "final model" claim is finalized.
- **No calibration or threshold-selection numbers exist yet** — the project cannot currently claim a validated decision threshold, only the sklearn default of 0.5.
- **The paired statistical significance test** between the top two Day 3 models (t-test / Wilcoxon) was never executed, so "model A is meaningfully better than model B" claims are not yet statistically supported.
- **A small, unexplained accuracy drift** (0.8527 vs 0.8524) was observed between two runs of the same Day 2 pipeline despite a fixed `random_state` — worth a dedicated reproducibility audit.
- **Day 1's 1-page PDF deliverable was never submitted** (only notebooks and CSVs), and the Day 1 README was left stale relative to the actual completed tasks.
- **One Day 3 write-up's recommendation directly contradicted its own results table** (claimed SelectKBest won when the printed numbers showed no-selection winning) — since corrected in this consolidated documentation, but the original notebook write-up should still be fixed at the source.
- **No subgroup / fairness analysis** (by sex, race, or age band) has been performed yet; given the sensitivity of income prediction, this is recommended before any production use.
- With more data or additional features (e.g. industry-level aggregates, geographic cost-of-living indices), there is likely room to improve beyond the current best F1 of 0.712.

## 14. How to Reproduce Training

```bash
# 1. Set up environment (see Section 15 for exact versions)
pip install -r requirements.txt

# 2. Run notebooks in order — each stage depends on the previous day's saved split/pipeline
jupyter nbconvert --to notebook --execute day1_eda_baselines.ipynb
jupyter nbconvert --to notebook --execute day2_preprocessing_baseline_models.ipynb
jupyter nbconvert --to notebook --execute day3_feature_engineering_cv.ipynb
jupyter nbconvert --to notebook --execute day4_tuning_calibration.ipynb
jupyter nbconvert --to notebook --execute day5_final_validation_interpretation.ipynb
```

- All `random_state` values are fixed (`random_state=42` throughout) — re-running should reproduce results, modulo the small drift noted in Section 13, which is still under investigation.
- The hold-out test set created on Day 1 is never touched by any tuning or model-selection step in Days 2–4; it is only scored against for final reporting.

## 15. How to Run Inference

```python
import joblib

# Load the saved pipeline (preprocessing + best available model)
pipeline = joblib.load("final_pipeline.joblib")

# new_data: a pandas DataFrame with the same raw columns as the training data
# — do NOT manually re-apply preprocessing; it's already inside the pipeline
predictions = pipeline.predict(new_data)
probabilities = pipeline.predict_proba(new_data)[:, 1]

# NOTE: threshold is currently the sklearn default (0.5) — this has not yet
# been validated against the project's precision-first business objective.
# Update this once Day 4's calibration/threshold step is completed.
THRESHOLD = 0.5
final_labels = (probabilities >= THRESHOLD).astype(int)
```

Tested against 5–10 new/unseen example rows per the Day 5 Task 4 requirement, confirming the pipeline runs end-to-end without any manual preprocessing step outside the saved artifact.

## 16. Environment / Library Versions

| Package | Role |
|---|---|
| Python | Base interpreter |
| pandas / numpy | Data loading, cleaning, array operations |
| scikit-learn | Preprocessing pipeline, models, cross-validation, tuning, calibration |
| matplotlib | ROC/PR curves, calibration plots, learning curves |
| joblib | Saving/loading the final fitted pipeline |
| scipy | Paired statistical test (t-test / Wilcoxon) |

Exact pinned version numbers are documented alongside the Day 4 saved pipeline (per that day's Task 1 requirement) — copy those verbatim into `requirements.txt` rather than re-typing from memory, to avoid a silent version mismatch between the environment that produced these results and the environment used to reproduce them.

## File Structure

```
├── Final_Project_Report_Day1-5.docx   # full consolidated report incl. mistakes log
├── README.md                          # this file
├── day1_eda_baselines.ipynb
├── day2_preprocessing_baseline_models.ipynb
├── day3_feature_engineering_cv.ipynb
├── day4_tuning_calibration.ipynb
├── day5_final_validation_interpretation.ipynb
├── final_pipeline.joblib              # saved preprocessing + best available model
└── requirements.txt
```
