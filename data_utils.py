import numpy as np
import pandas as pd

from sklearn.metrics import f1_score 
import sklearn.metrics as skm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from acs_helper import ACSData

def prep_real(real, verbose=False):
    X = real.iloc[:, :-1]
    y = real.iloc[:, -1]
    _, x_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return x_test, y_test

def prep_real_with_model(real, model, verbose=False):
    X = real.iloc[:, :-1]
    y = real.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_real = model() #(max_iter=1000)
    model_real.fit(x_train, y_train)

    #Test the model
    predictions = model_real.predict(x_test)
    f1 = f1_score(y_test, predictions)
    if verbose:
        print()
        print('Trained on Real Data')
        print(classification_report(y_test, predictions))
        print('Accuracy real: ' + str(accuracy_score(y_test, predictions)))
        print(f1)
    return x_test, y_test, f1

def get_scenario(scenario):
    if "ACSEmployment" == scenario:
        return get_employment()
    elif "ACSEmploymentAge" == scenario:
        return get_employment_large()
    elif "ACSMobility" == scenario:
        return get_mobility()
    elif "ACSPublicCoverage" == scenario:
        return get_public()
    elif "Mean" == scenario:
        return get_public()
    else:
        raise ValueError("Not supported: " + scenario)

def get_employment():
    acs = ACSData(states=['NY'])
    pd_all_data, pd_features, pd_target, pd_group = acs.return_simple_acs_data_scenario(scenario="ACSEmployment")
    return pd_all_data, pd_features, pd_target, pd_group

def get_employment_large():
    acs = ACSData(states=['CA'])
    pd_all_data, pd_features, pd_target, pd_group = acs.return_acs_data_scenario(scenario="ACSEmployment")
    return pd_all_data, pd_features, pd_target, pd_group

def get_mobility():
    acs = ACSData(states=["NM"])
    pd_all_data, pd_features, pd_target, pd_group = acs.return_simple_acs_data_scenario(scenario="ACSMobility")
    for col in pd_features.columns:
        codes = dict([(category, code) for code, category in enumerate(pd_all_data[col].unique())])
        pd_all_data[col] = pd_all_data[col].map(codes)
    pd_all_data = pd_all_data.drop("JWMNP", axis=1)
    pd_features = pd_features.drop("JWMNP", axis=1)
    pd_all_data = pd_all_data.drop("PINCP", axis=1)
    pd_features = pd_features.drop("PINCP", axis=1)
    pd_all_data = pd_all_data.drop("ESP", axis=1)
    pd_features = pd_features.drop("ESP", axis=1)

    return pd_all_data, pd_features, pd_target, pd_group

def get_public_class(state="NM"):
    acs = ACSData(states=[state])
    scenario = "ACSPublicCoverage"
    pd_all_data, pd_features, pd_target, pd_group = acs.return_simple_acs_data_scenario(scenario=scenario)
    pd_all_data = pd_all_data.drop("PINCP", axis=1)
    pd_features = pd_features.drop("PINCP", axis=1)
    for col in pd_features.columns:
        codes = dict([(category, code) for code, category in enumerate(pd_all_data[col].unique())])
        pd_all_data[col] = pd_all_data[col].map(codes)
    return pd_all_data, pd_features, pd_target, pd_group

def get_public(state="NY"):
    acs = ACSData(states=[state])
    return acs.acs_data

def calculate_dimensionality(df):
    dimensionality = 1
    for col in df.columns:
        unique_values = df[col].nunique()
        dimensionality *= unique_values
    return dimensionality

def force_data_categorical_to_numeric(df, cat_columns=[]):
    # convert columns to categorical if they are not already
    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
    return df