# Heart Failure Mortality Prediction (ANN)

> **Note:** Despite the repository name, this project has **nothing to do with cocktails**. It is a binary classification notebook that predicts heart failure mortality using an artificial neural network.

## Overview

A Jupyter notebook (`main_Code_blog.ipynb`) that builds and evaluates Keras/TensorFlow ANN models to predict whether a patient will experience a death event during follow-up, using clinical heart failure records.

## Dataset

**Heart Failure Clinical Records** — 12 clinical features per patient with a binary target `DEATH_EVENT`.

| Feature | Description |
|---------|-------------|
| `age` | Patient age |
| `anaemia` | Below-normal hemoglobin |
| `creatinine_phosphokinase` | CPK level (mcg/L) |
| `diabetes` | Diabetic status |
| `ejection_fraction` | Left ventricle ejection fraction |
| `high_blood_pressure` | Hypertension status |
| `platelets` | Platelet count (kilo platelets/mL) |
| `serum_creatinine` | Serum creatinine (mg/dL) |
| `serum_sodium` | Serum sodium (mEq/L) |
| `sex` | Patient sex |
| `smoking` | Smoking status |
| `time` | Follow-up period (months) |
| `DEATH_EVENT` | Target: patient deceased during follow-up |

Source: [Kaggle — Heart Failure Clinical Data](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)

## What the Notebook Does

1. Exploratory data analysis (correlation heatmap, distribution plots, box plots)
2. Feature/target split (`DEATH_EVENT` as target)
3. Train/test split with standard scaling
4. Sequential ANN with Dense, BatchNormalization, and Dropout layers
5. Experiments with different loss functions (`binary_crossentropy`, `categorical_crossentropy`, `mean_squared_error`) and activation functions
6. Evaluation with accuracy, precision, recall, F1, and confusion matrix

## Tech Stack

- Python 3
- pandas, numpy, seaborn, matplotlib
- scikit-learn (preprocessing, metrics, train_test_split)
- Keras / TensorFlow

## Prerequisites

```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow jupyter
```

Download the heart failure clinical dataset and place it where the notebook expects it.

## Usage

```bash
jupyter notebook main_Code_blog.ipynb
```

Run all cells to reproduce the EDA, model training, and evaluation results.
