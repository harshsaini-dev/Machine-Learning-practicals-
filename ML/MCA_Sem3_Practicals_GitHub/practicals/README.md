# MCA201P — Machine Learning Practical Files (Semester III)

**Student:** Harsh Saini  
**Roll Number:** 253028036  
**University:** COER University, Roorkee  
**Academic Session:** 2026–27

This directory contains the original **Lab Sheets 01–05** (35 experiments per sheet, 175 in total) plus **Lab Sheets 06–10** containing **10 mini-projects**. It is organized for practical demonstration and viva. The original Labs 01–05 are retained in the repository; this overlay ZIP only adds Labs 06–10 and refreshed documentation/dependencies.

> **Lab numbering:** Uploaded assignment filenames `MCA201P-AS07` through `AS11` contain printed **Lab Sheets 06–10**; these scripts are named according to the numbers *printed inside* the PDFs.

## Lab index

| Lab | Content | Python source | Jupyter notebook |
|:--:|---|---|---|
| 01 | Python setup and dataset loading: 35 experiments | `lab_01.py` | `lab_01.ipynb` |
| 02 | Data preprocessing: 35 experiments | `lab_02.py` | `lab_02.ipynb` |
| 03 | Regression models: 35 experiments | `lab_03.py` | `lab_03.ipynb` |
| 04 | Clustering and PCA: 35 experiments | `lab_04.py` | `lab_04.ipynb` |
| 05 | Reinforcement learning: 35 experiments | `lab_05.py` | `lab_05.ipynb` |
| 06 | Student performance & house price regression | `lab_06.py` | `lab_06.ipynb` |
| 07 | Customer segmentation & news-text clustering | `lab_07.py` | `lab_07.ipynb` |
| 08 | MNIST digit DNN & Q-learning maze solver | `lab_08.py` | `lab_08.ipynb` |
| 09 | COVID-19 analytics & sales analytics dashboards | `lab_09.py` | `lab_09.ipynb` |
| 10 | Loan approval & symptom-pattern classifiers | `lab_10.py` | `lab_10.ipynb` |

## Directory layout

```text
practicals/
├── README.md                     # updated project index and instructions
├── requirements.txt              # updated common dependencies
├── requirements_optional_tf.txt  # only needed to run MNIST DNN (Lab 08)
├── lab_01.py ... lab_10.py       # each lab's main script
├── lab_01.ipynb ... lab_10.ipynb # Jupyter files
├── generate_labs_09_10_data.py   # deterministic seed-based datasets
├── lab_10_app.py                 # optional Streamlit predictor UI
├── datasets/                     # input CSVs
├── outputs/lab_06 ... lab_10/    # charts, dashboard HTML, model artifacts, metrics
├── slides/                       # two 13-slide capstone project PPTX decks
├── tests/                        # automated project tests
├── reports/                      # original Labs 01–05 PDFs, already in repo
├── run_all.py                     # original Labs 01–05 runner
└── run_new_labs.py                # runner for new Labs 06–10
```

## Installation (Windows PowerShell)

From the `practicals` folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked, use `.venv\Scripts\python.exe` directly or `cmd.exe` with `.venv\Scripts\activate.bat`. Python **3.11 or 3.12** is recommended for broad scientific-package compatibility; select matching TensorFlow wheels if your Python version differs.

For Lab 08 MNIST only (TensorFlow installed separately because it is large):

```powershell
python -m pip install -r requirements_optional_tf.txt
```

## Run projects

```powershell
# Original lab scripts (already stored on GitHub):
python lab_01.py
python lab_02.py
python lab_03.py
python lab_04.py
python lab_05.py

# New mini-projects:
python lab_06.py
python lab_07.py
python lab_08.py   # requires TensorFlow and MNIST download on first run
python lab_09.py
python lab_10.py

# Execute new non-TensorFlow labs sequentially:
python run_new_labs.py

# Run with optional MNIST training too:
python run_new_labs.py --include-mnist

# Streamlit UI for loan and symptom-pattern prediction:
streamlit run lab_10_app.py

# Execute Jupyter notebooks:
python -m notebook

# Automated tests for Labs 09–10:
python -m unittest discover -s tests -v
```

The notebooks for Labs 09 and 10 include **executed outputs**. The `run_all.py` retained from your existing repository runs **only Labs 01–05**; `run_new_labs.py` adds Labs 06–10 without changing existing code.

## Data provenance and integrity

- **Labs 01–05:** Existing repository files include toy synthetic and built-in scientific datasets (e.g., Iris, diabetes and Linnerud), described in their original practical reports.
- **Lab 06:** Student and housing CSVs generated for classroom demonstration. **SYNTHETIC**.
- **Lab 07:** Customer records and short news snippets generated for classroom demonstration. **SYNTHETIC**.
- **Lab 08:** MNIST is a public, real digit image benchmark automatically fetched by Keras; the custom Q-learning maze is a simulation.
- **Lab 09:** COVID-like daily country data and business transaction records are **SYNTHETIC**. Dashboard counts are not genuine infection, recovery, mortality or vaccination records; the countries are named `Country A/B/C (simulated)`. Sales are not actual business financial statements.
- **Lab 10:** Loan and symptom-pattern records are **SYNTHETIC**. The trained models **must not** be used for real credit eligibility, healthcare triage or medical diagnosis. The UI prominently discloses these limitations.

Rebuild the Lab 09–10 CSV inputs with `python generate_labs_09_10_data.py` to reproduce the included training data. The random seed is fixed; numeric results may vary slightly with package versions.

## Lab 06 — Supervised learning

Student score regression uses attendance, internal marks, assignments, study hours and previous results. House price regression uses area, bedrooms, bathrooms, parking, age and location. Both:
- Use an **80/20 train/test split** (fixed random seed).
- Fit imputation/scaling/encoding through training-only pipelines.
- Compare Ridge and Random Forest regression using **MAE, RMSE and R²**.
- Save model pipelines with Joblib and actual-versus-predicted scatter plots to `outputs/lab_06/`.

## Lab 07 — Unsupervised learning

- Customer segmentation: K-Means and agglomerative clustering over scaled income, age, spending and purchase behavior; silhouette score for cluster-count selection; segmentation profiles and scatter plots.
- News article clustering: TF-IDF text features, K-Means and agglomerative clustering with cosine silhouette, SVD visual projection and an exported cluster-label CSV.
- Saved plots/models/metrics in `outputs/lab_07/`. Cluster IDs have no inherent meaning.

## Lab 08 — Deep learning and reinforcement learning

- **MNIST DNN:** dense ReLU network with Dropout and Softmax, 5 epochs, training/validation learning curves and held-out digit accuracy; saves `mnist_dnn.keras` once run.
- **Maze:** epsilon-greedy tabular Q-learning with step/goal penalties and an independent BFS shortest-path comparison; outputs Q-table and reward curve.
- Run Lab 08 locally with `requirements_optional_tf.txt` and internet access for initial MNIST dataset download. TensorFlow training was not verified in the original non-TensorFlow build environment.

## Lab 09 — Data analytics and visualization

- Run: `python lab_09.py`.
- COVID-like dashboard: time-series confirmed, recovered, deaths, vaccinations, new cases and country comparison.
- Sales dashboard: monthly revenue/profit, product/region metrics and unsupervised RFM (recency/frequency/monetary) customer segments.
- Interactive standalone dashboards: `outputs/lab_09/covid_dashboard.html` and `outputs/lab_09/sales_dashboard.html`. Open these in Chrome/Edge to filter/zoom/hover without starting a server.
- Exported screenshots: `outputs/lab_09/covid_dashboard_preview.png` and `outputs/lab_09/sales_dashboard_preview.png`. Also exports cleaned CSVs, summaries and `*_findings.json`.

## Lab 10 — Capstone applications

- Run: `python lab_10.py`; then `streamlit run lab_10_app.py`.
- Loan approval: categorical and numerical applicant features with stratified 80/20 split.
- Symptom-pattern classification: simulated condition-linked symptom indicators and duration, **not** a clinical diagnostic engine.
- Compare Logistic Regression and Random Forest, select by weighted F1, produce precision/recall/accuracy, confusion matrices, and ROC-AUC for binary loans.
- Save `*.joblib` preprocessing/model pipelines, `*_metrics.json`, model comparison CSV, and classification reports under `outputs/lab_10/`.
- Presentation files in `slides/`: two **13-slide** decks, including architecture/flow, methods, evaluation, limitations and viva guidance.
- Both predictions are educational only. Do not provide personal financial or patient data.

## Results and example observations

In the generated demonstration runs (your version may differ):
- Loan best estimator: **Logistic Regression**, weighted F1 ≈ **0.705** on synthetic held-out data.
- Symptom-pattern best estimator: **Random Forest**, weighted F1 ≈ **0.901** on synthetic held-out data.
- Lab 09 created a COVID simulation covering 3 simulated countries (450 daily records), and 2,000 synthetic sales transactions covering 200 customers.

**None of these synthetic test scores establish real-world predictive performance.** Load joblib files only from trusted sources, as pickle serialization can execute code. Do not commit confidential records, secrets or API keys.

## GitHub update instructions

**Target repository:** https://github.com/harshsaini-dev/Machine-Learning-practicals-  
**Target folder:** `ML/MCA_Sem3_Practicals_GitHub/practicals/`

After cloning this repository, copy/unzip the overlay contents directly **inside** the `practicals` folder, allowing the updated `README.md` and `requirements.txt` to replace the older versions. Keep the existing Labs 01–05 and `reports/` intact. Commit and push:

```powershell
git status
git add ML/MCA_Sem3_Practicals_GitHub/practicals
git commit -m "Add Labs 06-10 projects, dashboards and capstone demos; update docs/deps"
git push origin main
```

The GitHub connector may have read access without permission to push; if you receive an HTTP 403 error, grant **Contents: Read and write** permission for this repository to the GitHub app or perform the push locally.
