# FRT Project: Fast Random Tree-Wasserstein Distance for Time Series Classification

This project implements and evaluates the Fast Random Tree (FRT) algorithm for computing Tree-Wasserstein distances in time series classification tasks. The implementation includes both FRT and GOW (Generalized Optimal Warping) methods for comparison.

## Overview

The project focuses on time series classification using advanced distance metrics:
- **FRT (Fast Random Tree)**: A novel approach using hierarchically separated trees to compute Tree-Wasserstein distances efficiently
- **GOW (Generalized Optimal Warping)**: An optimal transport-based method for time series alignment

## Project Structure

```
FRTproject/
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
│   ├── gow/                     # GOW algorithm implementation
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

### GOW Algorithm (`src/gow/`)
- **Optimal Transport**: Sinkhorn algorithm for transport matrix computation
- **Function-based Warping**: Multiple monotonic functions for time alignment
- **Coordinate Descent**: Iterative optimization for weight vectors
- **Auto-scaling**: Automatic function scaling based on sequence lengths

### Datasets
- **UCR Time Series Archive**: Standard benchmark datasets for time series classification
- **Human Action Recognition**: 3D motion capture data for action classification
- **Multi-dimensional Time Series**: Support for various time series formats

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FRTproject
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
results = run_knn(X_train, y_train, X_test, y_test, alg="FRT")

# Run GOW-based KNN classification
results = run_knn(X_train, y_train, X_test, y_test, alg="GOW")
```

### FRT Pipeline

```python
from frt import run_frt_pipeline

# Run complete FRT pipeline
D_tr, D_te, meta = run_frt_pipeline(
    X_train, X_test,
    n_trees=16,
    time_weight="auto",
    random_state=123,
    level_edge_shift=1,
    n_jobs=-1
)
```

### GOW Distance

```python
from gow import gow_sinkhorn_autoscale
import ot

# Compute GOW distance between two sequences
C = ot.dist(sequence1, sequence2, metric="minkowski")
distance = gow_sinkhorn_autoscale([], [], C)
```

## Algorithm Details

### FRT (Fast Random Tree)
- Constructs 2-HST trees from time series data
- Uses weighted Euclidean distance with time scaling
- Computes Tree-Wasserstein distances in closed form
- Supports multiple trees for ensemble methods

### GOW (Generalized Optimal Warping)
- Implements optimal transport with Sinkhorn algorithm
- Uses coordinate descent for weight optimization
- Supports multiple warping functions (polynomial, exponential, logarithmic, etc.)
- Auto-scales functions based on sequence lengths

## File Descriptions

- `src/utilities.py`: Main pipeline and utility functions
- `src/frt/__init__.py`: FRT algorithm implementation
- `src/gow/__init__.py`: GOW algorithm implementation
- `src/test.ipynb`: Experimental notebook
- `results/`: results

## Contributing

This project is part of academic research on time series classification using optimal transport methods. For questions or contributions, please refer to the original research papers.

## License

This project is for academic research purposes. Please cite the original papers when using this code.

