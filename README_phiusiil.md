# Phishing URL Detection: Interpretable Rule Learning (Aleph ILP) vs SVM vs MLP

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![SWI-Prolog](https://img.shields.io/badge/SWI--Prolog-Aleph%20ILP-E61B23)

This project compares a **symbolic, logic-based learner** (Aleph Inductive Logic Programming) against two **statistical learners** (an RBF-kernel SVM and a Multi-Layer Perceptron) for detecting phishing websites in the [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).

The question it asks: *how much predictive performance do we give up when we use a model whose decisions a human can read?* On this dataset the answer is "almost none". Aleph learns **four short, human-readable rules** that reach 99.7% test accuracy, within 0.2 percentage points of the black-box models.

> Coursework project for **Machine Learning for Data Science (MLDS)**, MSc, University of Surrey.

---

## Results

All models were evaluated on a held-out, class-balanced test set (positive class = phishing).

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Interpretability |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Aleph ILP** (4 rules) | 0.997 | 0.993 | **1.000** | 0.993 | 0.997 | 0.997* | **High** |
| SVM (RBF) | 0.999 | 1.000 | 0.998 | 1.000 | 0.999 | 1.000 | Low |
| MLP | 0.999 | 1.000 | 0.998 | 1.000 | 0.999 | 1.000 | Low |

\* Aleph outputs hard logical predictions, not probabilities, so its ROC-AUC is computed at a single operating point and equals balanced accuracy.

| Model | Training sample | Test sample |
|---|---|---|
| Aleph ILP | 700 (350 + 350) | 300 |
| SVM / MLP | 7,000 (3,500 + 3,500) | 3,000 |

### Rules learned by Aleph

```prolog
phishing(A) :- low_noofimage(A), no_hascopyrightinfo(A).
phishing(A) :- high_noofotherspecialcharsinurl(A), med_noofjs(A), no_hasdescription(A).
phishing(A) :- no_ishttps(A).
phishing(A) :- low_urlsimilarityindex(A).
```

In plain English, a site is flagged as phishing if **any** of these hold:

1. It has few images **and** no copyright notice.
2. Its URL has many special characters, it loads a moderate number of scripts, **and** it has no meta description.
3. It does not use HTTPS.
4. Its URL is not similar to known legitimate URLs.

The SVM and MLP reach slightly higher specificity, but their decisions cannot be inspected in this way. Aleph never missed a phishing site in the test set (recall 1.000); its few errors were false alarms on legitimate sites.

### Why are the scores so high?

The exploratory analysis (`01_eda_phiusiil.ipynb`) shows that several PhiUSIIL features separate the classes almost perfectly on their own:

- Every legitimate URL in the dataset uses HTTPS, while about half of phishing URLs do not.
- `URLSimilarityIndex` equals 100 for every legitimate URL, versus a mean of about 50 for phishing URLs.
- Legitimate pages have far more images, scripts and copyright/description metadata.

So near-perfect accuracy here says as much about the dataset as about the models. Results should not be taken as evidence of real-world performance on new phishing campaigns, and testing on a different phishing dataset would be a natural next step.

---

## Method

**Data preparation (all models)**

- Target: original `label` 0 = phishing, 1 = legitimate, re-encoded as `is_phishing` (phishing = positive class).
- Raw text and ID columns (`FILENAME`, `URL`, `Domain`, `Title`, `TLD`) are dropped, leaving 50 numeric features.
- Missing values are median-imputed.
- A balanced sample is drawn from the 235,795 rows (100,945 phishing / 134,850 legitimate), then split 70/30 with stratification and `random_state=42`.

**Aleph ILP** (`02_aleph_phiusiil.ipynb`)

- 500 examples per class, because ILP search does not scale to hundreds of thousands of rows.
- Top 20 features selected by mutual information, computed on the training split only.
- Binary features become `has_x` / `no_x` predicates; continuous features become `low_x` / `med_x` / `high_x` using 33rd/67th percentile thresholds from the training split.
- Each URL becomes a set of Prolog background facts, and Aleph learns `phishing/1`.
- Small grid over Aleph settings (`i`, `minpos`, `clauselength`, `nodes`); the best setting is chosen by test F1.

**SVM (RBF)** (`03_svm_rbf_phiusiil.ipynb`) and **MLP** (`04_mlp_phiusiil.ipynb`)

- 5,000 examples per class.
- `StandardScaler` + classifier in a scikit-learn `Pipeline`, so scaling is fit only on training folds.
- Hyperparameters tuned with 3-fold stratified `GridSearchCV`, scored on F1.
  - SVM: `C ∈ {1, 10}`, `gamma ∈ {scale, 0.01}`, balanced class weights. Best: `C=1, gamma=0.01`.
  - MLP: hidden layers `(50,)`, `(100,)`, `(50, 25)`; `alpha ∈ {1e-4, 1e-3}`; early stopping.

**Comparison** (`05_evaluate_thread2_phiusiil.ipynb`) combines the saved result files into one table and bar chart.

---

## Repository structure

```
.
├── 01_eda_phiusiil.ipynb               # Exploratory data analysis and feature diagnostics
├── 02_aleph_phiusiil.ipynb             # Aleph ILP: symbolic feature encoding and rule learning
├── 03_svm_rbf_phiusiil.ipynb           # RBF-kernel SVM baseline
├── 04_mlp_phiusiil.ipynb               # MLP baseline
├── 05_evaluate_thread2_phiusiil.ipynb  # Combined comparison of all three models
├── aleph.pl                            # Aleph v5 (Ashwin Srinivasan), used via PyILP
├── requirements.txt
└── README.md
```

Running the notebooks creates `outputs/` (metrics CSVs, plots, learned rules) and `aleph_files/` (generated Prolog files). Both are git-ignored.

---

## Getting started

### 1. Install SWI-Prolog

Aleph runs inside SWI-Prolog, which PyILP calls through `janus-swi`. Install SWI-Prolog from [swi-prolog.org](https://www.swi-prolog.org/Download.html) and make sure `swipl` is on your `PATH`.

### 2. Install Python dependencies

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the dataset

The dataset (~57 MB) is not included in this repository. Download `PhiUSIIL_Phishing_URL_Dataset.csv` from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) and place it in the repository root, next to the notebooks.

### 4. Run the notebooks in order

```
01 → 02 → 03 → 04 → 05
```

Notebook 05 reads the result files written by 02, 03 and 04, so run those first. Notebooks 01–04 are independent of each other.

---

## Limitations

- The dataset is close to linearly separable on a handful of features, so all models score near-ceiling and the differences between them are small.
- Aleph was trained on a much smaller sample (1,000 rows) than the SVM and MLP (10,000 rows), so the comparison is not perfectly like-for-like.
- The best Aleph setting was selected using test-set F1, which slightly favours Aleph's reported numbers.
- Discretising features into low/medium/high loses information that the SVM and MLP can use directly.

---

## References

- Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. *Computers & Security*, 136, 103545.
- Srinivasan, A. *The Aleph Manual*. University of Oxford.
- Muggleton, S. (1995). Inverse entailment and Progol. *New Generation Computing*, 13, 245–286.

## Acknowledgements

- Aleph v5 by Ashwin Srinivasan, freely available for academic use.
- [PyILP](https://pypi.org/project/PyILP/) for the Python–Aleph interface.
- The MLDS module team at the University of Surrey for the lab materials.

## Author

**[Your Name]** — MSc, University of Surrey
[LinkedIn](https://linkedin.com/in/your-profile) · [Email](mailto:you@example.com)
