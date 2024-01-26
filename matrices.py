# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime, ACSIncomePovertyRatio
import folktables
import os
import numpy as np

pd.set_option('display.max_columns', None)  

# %% [markdown]
# ### ACS Datasets

# %%
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["CA", "OR", "WA"], download=True) # , "NV", "AZ"

ACSHealthInsurance = folktables.BasicProblem(
    features=[
        'AGEP',
        'SCHL',
        'MAR',
        'SEX',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'RACAIAN',
        'RACASN',
        'RACBLK',
        'RACNH',
        'RACPI',
        'RACSOR',
        'RACWHT',
        'PINCP',
        'ESR',
        'ST',
        'FER',
        'RAC1P',
    ],
    target='HINS2',
    target_transform=lambda x: x == 1,
    group='RAC1P',
    preprocess=lambda x: x,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

folktables = {
    # "ACSEmployment": ACSEmployment,
    # "ACSIncome": ACSIncome,
    # "ACSMobility": ACSMobility,
    # "ACSPublicCoverage": ACSPublicCoverage,
    # "ACSTravelTime": ACSTravelTime,
    "ACSInsurance": ACSHealthInsurance,
    # "ACSPoverty": ACSIncomePovertyRatio
}

#race_agg_names = {1: 'White',
#                  2: 'Black or African American alone',
#                  3: 'Asian alone',
#                  4: 'Other'}

RAC1P_mapper = {1:1,
                2:2,
                3:4,
                4:4,
                5:4,
                6:3,
                7:4,
                8:4,
                9:4}

#SEX_mapper = {1: 1,
#              2: 2}

#HISP_mapper = {x: 1 if x == 0 else 2 for x in range(0, 24)}

# New codes:
# 1 - White alone
# 2 - Black or African American alone
# 3 - Asian alone
# 4 - Other

for name in list(folktables.keys()):
    df = None
    
    # Add HISP
    folktables[name].features.append('HISP')
    
    features, label, group = folktables[name].df_to_numpy(acs_data)
    feature_names = folktables[name].features
    df = pd.DataFrame(features, columns = feature_names)
    df['RAC1P_recoded'] = df['RAC1P'].map(RAC1P_mapper)
    df['label'] = label

    outdir = f'matrices/{name}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        #os.chmod(outdir,rwx)

    # Save full datasets
    X = df.drop(columns=['label'])
    X.to_csv(f'{outdir}/X.csv',index=False)

    y = df['label'].apply(lambda x: 1 if x else 0)
    y.to_csv(f'{outdir}/y.csv',index=False)

    # Save sample datasets
    _ , dfs = train_test_split(df, test_size=0.05,random_state=42)
    dfs.reset_index(inplace=True,drop=True)
    
    X = dfs.drop(columns=['label'])
    X.to_csv(f'{outdir}/Xs.csv',index=False)

    y = dfs['label'].apply(lambda x: 1 if x else 0)
    y.to_csv(f'{outdir}/ys.csv',index=False)

for name in list(folktables.keys()):
    df = None
    
    # Add HISP
    folktables[name].features.append('HISP')
    
    features, label, group = folktables[name].df_to_numpy(acs_data)
    feature_names = folktables[name].features
    df = pd.DataFrame(features, columns=feature_names)
    df['RAC1P_recoded'] = df['RAC1P'].map(RAC1P_mapper)
    df['label'] = label

    # Filter for RAC1P_recoded = 1 or 2
    df_filtered = df[(df['RAC1P_recoded'] == 1) | (df['RAC1P_recoded'] == 2)]

    outdir = f'matrices/{name}_mfopt'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Save full filtered datasets
    X_filtered = df_filtered.drop(columns=['label'])
    X_filtered.to_csv(f'{outdir}/X.csv', index=False)

    y_filtered = df_filtered['label'].apply(lambda x: 1 if x else 0)
    y_filtered.to_csv(f'{outdir}/y.csv', index=False)

    # Save sample filtered datasets
    _ , dfs = train_test_split(df_filtered, test_size=0.05, random_state=42)
    dfs.reset_index(inplace=True, drop=True)
    
    Xs_filtered = dfs.drop(columns=['label'])
    Xs_filtered.to_csv(f'{outdir}/Xs.csv', index=False)

    ys_filtered = dfs['label'].apply(lambda x: 1 if x else 0)
    ys_filtered.to_csv(f'{outdir}/ys.csv', index=False)



# %% [markdown]
# ### Portuguese Students

# %%
students_df_raw = pd.read_csv('data/students.csv',delimiter=',')
students_df = students_df_raw.copy()

# Pre-processing: make target, code sex, address, parents education
students_df['label'] = students_df['G3'].apply(lambda x: 1 if x < 10 else 0)
students_df['sex'] = students_df['sex'].apply(lambda x: 1 if x == 'F' else 0)
students_df['address'] = students_df['address'].apply(lambda x: 1 if x == 'R' else 0)

# Mapping for Education
# 1: Other
# 2: High school
# 3: University or greater
students_df['parents_education'] = students_df.apply(lambda row: max(row.Medu,row.Fedu), axis=1)

# From the Portuguese students dataset:
# (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)

students_education_recoder = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 2,
}

students_df['parents_education'] = students_df['parents_education'].map(students_education_recoder)

def recode_to_binary(x):
    if x == 'yes':
        return 1
    elif x == 'no':
        return 0
    else:
        return x

students_df = students_df.applymap(recode_to_binary)

one_hot_variables = ['Subject','school','famsize','Pstatus','Mjob','Fjob','reason','guardian']
for v in one_hot_variables:
    temp = None # This is likely useless, but I have a reason for keeping it... it will remain a mystery to you, the reader of this code
    temp = pd.get_dummies(students_df[v])
    students_df = pd.merge(students_df,temp.add_suffix(f'_{v}'), how='left',left_index=True, right_index=True)
    students_df.drop(columns=[v],inplace=True)

students_df.drop(columns=['ID'],inplace=True)

outdir = f'matrices/students'
if not os.path.exists(outdir):
    os.makedirs(outdir)
    #os.chmod(outdir,rwx)

X = students_df.drop(columns=['label'])
X.to_csv(f'{outdir}/X.csv',index=False)

y = students_df['label'].apply(lambda x: 1 if x else 0)
y.to_csv(f'{outdir}/y.csv',index=False)

X.head()

# %% [markdown]
# ### Tiawenese Loan Assessment

# %%
loans_df_raw = pd.read_csv('data/loans.csv',delimiter=',')
loans_df = loans_df_raw.copy()
loans_df['sex'] = loans_df['SEX'].apply(lambda x: 1 if x == 2 else 0)
loans_df.drop(columns=['SEX'],inplace=True)

# Mapping for Education
# 1: Other
# 2: High school
# 3: University or greater
loans_df['education'] = loans_df['EDUCATION']

# From the dataset description:
# Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)

loans_education_recoder = {
    1: 2,
    2: 2,
    3: 1,
    4: 0,
}

loans_df['education'] = loans_df['education'].map(loans_education_recoder)
loans_df['education'] = loans_df['education'].fillna(0)

loans_df['label'] = loans_df['default payment next month']

outdir = f'matrices/loans'
if not os.path.exists(outdir):
    os.makedirs(outdir)
    #os.chmod(outdir,rwx)

loans_df.drop(columns='ID',inplace=True)

X = loans_df.drop(columns=['label'])
X.to_csv(f'{outdir}/X.csv',index=False)

y = loans_df['label'].apply(lambda x: 1 if x else 0)
y.to_csv(f'{outdir}/y.csv',index=False)

X.head()

# %% [markdown]
# ### Diabetes

# %%
diabetes_df_raw = pd.read_csv('data/diabetes.csv',delimiter=',')
diabetes_df = diabetes_df_raw.copy()

# Pre-processing: make target, code sex, address, parents education
diabetes_df['label'] = diabetes_df['class'].apply(lambda x: 1 if x == 'Positive' else 0)
diabetes_df.drop(columns=['class'],inplace=True)

diabetes_df['SEX'] = diabetes_df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
diabetes_df.drop(columns=['Gender'],inplace=True)

def recode_to_binary(x):
    if x == 'Yes':
        return 1
    elif x == 'No':
        return 0
    else:
        return x

diabetes_df = diabetes_df.applymap(recode_to_binary)

outdir = f'matrices/diabetes'
if not os.path.exists(outdir):
    os.makedirs(outdir)
    #os.chmod(outdir,rwx)

X = diabetes_df.drop(columns=['label'])
X.to_csv(f'{outdir}/X.csv',index=False)

y = diabetes_df['label'].apply(lambda x: 1 if x else 0)
y.to_csv(f'{outdir}/y.csv',index=False)

X.head()

# %% [markdown]
# ### Heart Disease

# %%
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
heart_disease_df = heart_disease.data.features 
heart_disease_df['label'] = heart_disease.data.targets

outdir = f'matrices/heart_disease'
if not os.path.exists(outdir):
    os.makedirs(outdir)
    #os.chmod(outdir,rwx)

X = heart_disease_df.drop(columns=['label'])
X.to_csv(f'{outdir}/X.csv',index=False)

y = heart_disease_df['label'].apply(lambda x: 1 if x else 0)
y.to_csv(f'{outdir}/y.csv',index=False)

X.head()

# %%



