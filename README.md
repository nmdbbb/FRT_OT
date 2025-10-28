# FRT_OT: Fast Random Tree-Wasserstein Distance for Time Series Classification

This project implements and evaluates the Fast Random Tree (FRT) algorithm for computing Tree-Wasserstein distances in time series classification tasks.

Repository: `https://github.com/nmdbbb/FRT_OT.git`

## Overview

The project focuses on time series classification using advanced distance metrics:
- **FRT (Fast Random Tree)**: A novel approach using hierarchically separated trees to compute Tree-Wasserstein distances efficiently

## Project Structure

```
FRT_OT/
├── data/                          # Datasets
│   ├── Human_Actions/            # Human action recognition datasets
│   │   ├── MSRAction3D/         # Microsoft Research Action3D dataset
│   │   ├── MSRDailyActivity3D/  # Daily activity dataset
│   │   ├── SpokenArabicDigit/   # Arabic digit recognition
│   │   └── Weizmann/            # Weizmann action dataset
│   └── UCR/                      # UCR Time Series Archive datasets
│       ├── BasicMotions/        # Basic motion patterns
│       ├── BME/                 # Biomedical signals
│       ├── Chinatown/           # Traffic data
│       ├── DistalPhalanxTW/     # Medical imaging
│       └── ItalyPowerDemand/    # Power consumption data
├── src/                          # Source code
│   ├── frt/                     # FRT algorithm implementation
│   ├── utilities.py             # Utility functions and main pipeline
│   └── test.ipynb               # Jupyter notebook for experiments
├── results/                      # Experimental results
│   └── knn_frt_results.csv      # KNN classification results
└── requirements.txt             # Python dependencies
```

## Key Features

### FRT Algorithm (`src/frt/`)
- **Hierarchically Separated Trees (2-HST)**: Deterministic-friendly tree construction
- **Auto Time Weighting**: Median ratio heuristic for optimal time scaling
- **Closed-form Tree-Wasserstein**: Efficient distance computation
- **Global Pipeline**: Unified training and testing distance matrix computation

### Datasets
- **UCR Time Series Archive**: Standard benchmark datasets for time series classification
- **Human Action Recognition**: 3D motion capture data for action classification
- **Multi-dimensional Time Series**: Support for various time series formats

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nmdbbb/FRT_OT.git
cd FRT_OT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

The main functionality is available through the `utilities.py` module:

```python
from utilities import run_knn, load_ucr_dataset, load_human_action_dataset

# Load UCR dataset
X_train, y_train, X_test, y_test = load_ucr_dataset("data/UCR", "BasicMotions")

# Run FRT-based KNN classification
results, knn_secs = run_knn(X_train, y_train, X_test, y_test, alg="FRT")
```

### FRT Pipeline

```python
from frt import frt_knn

# Run complete FRT pipeline
D_tr, D_te, meta = frt_knn(
    X_train, X_test,
    n_trees=16,
    time_weight="auto",
    time_factor=64.0,
    depth_shift="auto",
    level_edge_shift=1,
    random_state=123,
)
```

<!-- Additional algorithms intentionally omitted in this README revision -->

## Algorithm Details

### FRT (Fast Random Tree)
- Constructs 2-HST trees from time series data
- Uses weighted Euclidean distance with time scaling
- Computes Tree-Wasserstein distances in closed form
- Supports multiple trees for ensemble methods

## File Descriptions

- `src/utilities.py`: Main pipeline and utility functions
- `src/frt/__init__.py`: FRT algorithm implementation
- `src/test.ipynb`: Experimental notebook
- `results/`: results

## Extending: Plug in a new distance algorithm

You can add another distance algorithm alongside FRT by extending `run_knn` in `src/utilities.py`.

1) Implement or import your distance function to produce a test–train distance matrix `D_te` and (optionally) a train–train matrix `D_tr`.

2) Add a new branch in `run_knn`:

```python
# inside utilities.run_knn(...)
elif alg == "MY_ALG":
    # Prepare any parameters and precomputations here
    # Build D_tr (optional, used when metric="precomputed")
    # Build D_te (shape: n_test x n_train)
    D_tr = ...  # square (n_train x n_train)
    D_te = ...  # rectangular (n_test x n_train)
```

3) Keep the KNN call unchanged, since it uses the precomputed metric:

```python
from sklearn import neighbors
from sklearn.metrics import accuracy_score
clf = neighbors.KNeighborsClassifier(n_neighbors=k_actual, metric="precomputed")
clf.fit(D_tr, y_train)
acc = accuracy_score(y_test, clf.predict(D_te))
```

Notes:
- Normalize distances (e.g., divide by `max(D_tr)`) for stability.
- Ensure `D_te` and `D_tr` are non-negative and symmetric (`D_tr`) as required.
- For efficiency, prefer vectorized computation and avoid nested Python loops when possible.

## Contributing

This project is part of academic research on time series classification using optimal transport methods. For questions or contributions, please refer to the original research papers.

## License

This project is for academic research purposes. Please cite the original papers when using this code.

