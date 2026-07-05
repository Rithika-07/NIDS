# NIDS Project

This project implements a Network Intrusion Detection System (NIDS) using the UNSW-NB15 dataset. It includes both binary (Normal vs. Anomaly) and multi-class (Attack Category) classifications using a Random Forest Classifier.

## Prerequisites

- Python 3.x
- Virtual environment (recommended)

## Setup and Installation

1. Clone or download the repository.
2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
   ```
4. Ensure the dataset `UNSW_NB15_training-set.csv` is present in the root directory.

## Running the Project

You can run the network intrusion detection system model training and evaluation script:

```bash
python run_nids.py
```

This will:
- Load the dataset.
- Train and evaluate a binary classification model (Normal vs. Anomaly).
- Train and evaluate a multi-class classification model (Categorizing different attack types).
- Generate visualizations (e.g., `attack_distribution.png`).
- Save the trained models and encoders (`*.pkl` files).

Alternatively, you can explore the data and model interactively using the provided Jupyter notebook:
```bash
jupyter notebook NIDS.ipynb
```
