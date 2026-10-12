# MCA201P - MCA Semester III Practicals

**Student:** Harsh Saini  
**Roll number:** 253028036  
**University:** COER University, Roorkee  
**Session:** 2026-2027

Five lab sheets, with all 35 experiments in each sheet (175 total).
The notebooks contain executed code cells, captured console output, and graphs.
The PDFs use a simple practical-file format: aim, procedure, source code,
output, observation, and conclusion.

## Files

| Lab | Topic | Notebook | Script |
|---|---|---|---|
| 01 | Python setup and dataset loading | `lab_01.ipynb` | `lab_01.py` |
| 02 | Data preprocessing | `lab_02.ipynb` | `lab_02.py` |
| 03 | Regression models | `lab_03.ipynb` | `lab_03.py` |
| 04 | Clustering and PCA | `lab_04.ipynb` | `lab_04.py` |
| 05 | Reinforcement learning | `lab_05.ipynb` | `lab_05.py` |

- `reports/`: the five separate practical PDFs.
- `datasets/`: all five input CSV files; no dataset download is needed.
- `outputs/`: actual graphs, processed CSV files, and trained models.
- `outputs/lab_XX/console_outputs.txt`: full captured text for all experiments.
- `environment.txt`: versions used to produce the included results.

## Install on Windows

Extract the ZIP. Open a terminal in the folder containing this README.

```powershell
python --version
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m notebook
```

If PowerShell blocks activation, use Command Prompt and
`.venv\Scripts\activate.bat`, or call `.venv\Scripts\python.exe` directly.

Open a notebook and run the common setup first, then Programs 1-35 in order.
Later experiments reuse variables and helper functions from earlier ones.
Each experiment has its own code cell. Avoid running individual cells out of order.

## Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook
```

## Run as scripts

```bash
python lab_01.py
python lab_02.py
python lab_03.py
python lab_04.py
python lab_05.py
```

Plot windows may need to be closed to continue a script. To run every sheet
without opening plot windows:

```bash
python run_all.py
```

The reinforcement-learning script trains actual agents and repeats controlled
experiments. Runtime depends on the computer. It runs on CPU; a GPU is not required.

## Datasets and environments

- `students.csv`: 12 illustrative student records prepared for Pandas practice;
  includes two missing cells and a duplicate.
- `employees.csv`: 20 illustrative records prepared for preprocessing; includes
  missing values, a sparse bonus column, a salary outlier, dates, and a target.
  These two small teaching datasets are synthetic, not real personal data.
- `diabetes.csv`: the real scikit-learn diabetes dataset (442 records).
  The bundled inputs are already scaled by the dataset provider. Target:
  quantitative disease progression one year after baseline.
- `linnerud.csv`: the real scikit-learn Linnerud exercise dataset (20 records).
- `iris.csv`: the real scikit-learn Iris measurements, without species labels.
- FrozenLake-v1: non-slippery 4x4 map for a reproducible first Q-learning example.
- Grid World: custom 4x4 Gymnasium environment with walls, a -0.01 step penalty,
  and a goal reward of 1; maximum 64 steps.
- CartPole-v1: PyTorch DQN and discretized tabular Q-learning; maximum 500 steps.

Official dataset documentation:
- https://scikit-learn.org/stable/datasets/toy_dataset.html
- https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- https://gymnasium.farama.org/environments/classic_control/cart_pole/
- https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

## Reading the results

All included outputs and plots come from executed programs. They are not typed
sample predictions. Setup commands tell you what to run locally; printed
versions describe the environment used to generate these files.
Jupyter Notebook was installed and its server launch was checked on localhost.
IDE-specific buttons are not shown as if they had been clicked in VS Code or PyCharm.

- Lab 02 splits the original data before fitting the final predictive
  preprocessing pipeline. Earlier full-data preprocessing examples are
  descriptive demonstrations, not model-evaluation pipelines.
- Lab 03 compares regression models on the same train/test indices. Polynomial
  models use BMI alone; multiple regression uses ten features. Linnerud has only
  five held-out rows and its measured R2 is negative; this is retained honestly.
- Lab 04 avoids treating silhouettes in different feature spaces as directly
  comparable. A common-space score and adjusted Rand index are also printed.
- Lab 05 includes a same-environment CartPole comparison, actual step counts,
  matched evaluation seeds, two-seed ablations, and learning-rate experiments.
  Fixed episode budgets and few seeds are classroom demonstrations, not a
  benchmark claiming that one algorithm is universally better. Training
  exploration differs from greedy evaluation. True termination stops the
  bootstrap; time-limit truncation only ends the rollout.
- Exact library versions, execution time, and floating-point results can differ
  on another system. Fixed seeds help reproducibility but do not remove all
  platform differences.

Only load the provided model files if you trust their source. Joblib/pickle
model loading can execute code; the PyTorch checkpoint uses `weights_only=True`.

## Upload using your existing Git setup

Create an empty GitHub repository named `mca-sem3-practicals`, then copy its URL.
Run these commands from the extracted folder, replacing `YOUR_REPOSITORY_URL`:

```bash
git init
git branch -M main
git add .
git commit -m "Add MCA Semester III practicals - five complete lab sheets"
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

If this folder is already inside your existing practical repository, skip
`git init`, `git branch`, and `git remote add`; commit and push the new files
from that repository. If `origin` already exists, check it with
`git remote -v` before changing anything.

For a QR code in the printed practical file, use the repository's final HTTPS
URL after upload. Open the QR destination on a phone to check it before printing.
