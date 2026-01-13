# Ransomware Detection Using Machine Learning
## Project Overview

This project investigates the use of machine learning techniques for ransomware detection using the UGRansomware dataset. The objective is to train and evaluate classification models capable of identifying ransomware-related activity based on network, behavioural, and transactional features extracted from the dataset.

The project was developed as part of a university coursework assignment and focuses on model performance, robustness, and comparative analysis rather than real-time deployment.

## Objectives

The main objectives of this project are:

To build a baseline machine learning model for ransomware detection.

To evaluate whether semi-supervised learning can improve classification performance.

To analyse the impact of removing financial features on detection accuracy.

To compare the performance of Random Forest and XGBoost classifiers.

## Dataset

The project uses the UGRansomware dataset, which contains labelled instances of ransomware-related activity along with multiple feature types, including:

Network-level attributes

Behavioural indicators

Financial and transactional features

The dataset is provided as a CSV file and should be placed in the data/directory.

## File Structure
AIvirustool/
│
├── data/
│   └── data.csv
│
├── outputs/
│   ├── run_summary.json
│   ├── feature_importance.csv
│   ├── xgboost_best_params.json
│   └── plots/
│       └── class_distribution.png
│
├── notebooks/
│   └── analysis.ipynb
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

## How to run
pip install -r requirements.txt
python main.py
Run all experiments (recommended for full results):
python main.py --save-plots --save-feature-importance --run-semi-supervised --run-no-financials --run-xgboost --tune-xgboost

## Outputs

Running the script generates the following outputs:

Model accuracy summaries (run_summary.json)

Feature importance rankings (feature_importance.csv)

Tuned model parameters for XGBoost (xgboost_best_params.json)

Exploratory data analysis plots

All outputs are saved to the outputs/ directory.
