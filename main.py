import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

def loadAndClean(filepath: str) -> pd.dataFrame:
    df = pd.read_csv('data.csv')

    df = df[df['Time'] >= 0]
    df = df[df['ExpAddress'] != '1']
    df.rename(columns={'Protcol': 'Protocol'}, inplace=True)

    df.reset_index(drop=True, inplace=True)

    

