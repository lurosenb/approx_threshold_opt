import numpy as np
import pandas as pd
import os


from sklearn.metrics import f1_score 
import sklearn.metrics as skm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from acs_helper import ACSData

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

def concat_results(directory="results", output_file="all_results.pkl"):
    pkl_files = [file for file in os.listdir(directory) if file.endswith('.pkl')]
    df_list = [pd.read_pickle(os.path.join(directory, file)) for file in pkl_files]
    big_dataframe = pd.concat(df_list, ignore_index=True)
    big_dataframe.to_pickle(directory + '/' + output_file)