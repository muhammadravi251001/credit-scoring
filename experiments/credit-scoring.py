#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# Optional: Install and load the 'woe' extension
# !pip install woe
# %load_ext woe

# Optional: Install and load the 'scorecardpy' extension
# !pip install scorecardpy
# %load_ext scorecardpy

# Optional: Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Define file path and NA values
file_path = "/content/drive/MyDrive/Drive BFR/KASDD/final_lending_club_loans.csv"
na_values = ['#VALUE!', '#DIV/0!']

# Read dataset
dataset = pd.read_csv(file_path, na_values=na_values)

# Display the first 5 rows of the dataset
dataset.head(5)

# Display the shape of the dataset
dataset.shape

# Explore the dataset
dataset.info()

# Describe numeric attributes
dataset.describe()

# Describe non-numeric attributes
dataset.describe(include=np.object)

# Count unique elements in the dataset
dataset[dataset.columns].nunique()

# Calculate the correlation matrix for numeric attributes
dataset.corr(method='pearson')

# Visualize boxplots for each attribute to check for outliers
dataset.boxplot(figsize=(20, 3))

# Note: You may uncomment the optional code for mounting Google Drive if needed.

# Visualize heatmap for dataset
fig, ax = plt.subplots(figsize=(16, 16))

corrmat = dataset.corr()
sns.heatmap(corrmat, annot=True, square=True, ax=ax)

# Visualize histograms to check the distribution of values in each column
dataset.plot.hist(dataset.columns.tolist(), bins=12, alpha=0.5)

# Check for duplicate data
dataset.duplicated(keep=False).sum()

# Function to check for missing values
def check_null(df):
    col_na = df.isnull().sum().sort_values(ascending=False)
    percent = col_na / len(df)

    missing_data = pd.concat([col_na, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data[missing_data['Total'] > 0])

# Check for missing values in the dataset
check_null(dataset)

# Fill missing values with the mean
dataset.fillna(dataset.mean(), inplace=True)

# Function to check for outliers
def check_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    print(((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum())

# Check for outliers in the dataset
check_outliers(dataset)

# Function to delete outlier rows
def delete_outlier(dataframe):
    q1 = dataframe.quantile(0.25)
    q3 = dataframe.quantile(0.75)
    iqr = q3 - q1
    return dataframe[~((dataframe < (q1 - 1.5 * iqr)) | (dataframe > (q3 + 1.5 * iqr))).any(axis=1)]

# Handle outliers by dropping outlier rows
dataset = delete_outlier(dataset)

# Drop redundant and forward-looking columns
dataset.drop(columns=['id', 'member_id', 'zip_code', 'total_rec_prncp'], inplace=True)

# View information about the target variable
dataset['loan_status'].value_counts()

# Conversion based on rules
conversion = {"loan_status": {"Fully Paid": 1, "Charged Off": 0}}

# Replace data according to conversion rules
dataset = dataset.replace(conversion)

# View information about the target variable
dataset['loan_status'].value_counts()

# View dataset information
dataset.info()

# Split the data into a 20% test size
X = dataset.drop('loan_status', axis=1)
y = dataset['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Function to convert emp_length to numeric
def emp_length_converter(df, column):
    df[column] = df[column].str.replace('\+ years', '')
    df[column] = df[column].str.replace('< 1 year', str(0))
    df[column] = df[column].str.replace(' years', '')
    df[column] = df[column].str.replace(' year', '')
    df[column] = pd.to_numeric(df[column])
    df[column].fillna(value=0, inplace=True)

# Convert emp_length information
emp_length_converter(X_train, "emp_length")
emp_length_converter(X_test, "emp_length")

# Function to convert term to numeric by removing "months"
def loan_term_converter(df, column):
    df[column] = pd.to_numeric(df[column].str.replace(' months', ''))

# Convert term information
loan_term_converter(X_train, "term")
loan_term_converter(X_test, "term")

# Function to replace comma with a period and convert to numeric
def comma_to_point_converter(df, column):
    df[column] = pd.to_numeric(df[column].str.replace(',', '.'))

# Convert installment information
comma_to_point_converter(X_train, "installment")
comma_to_point_converter(X_test, "installment")

# Remove dots in the int_rate feature
X_train['int_rate'] = X_train.int_rate.str.replace(',', '.')
X_test['int_rate'] = X_test.int_rate.str.replace(',', '.')

# Remove percent signs in the int_rate feature
X_train['int_rate'] = X_train.int_rate.str.replace('%', '')
X_test['int_rate'] = X_test.int_rate.str.replace('%', '')

# Convert the int_rate feature to numeric
X_train['int_rate'] = pd.to_numeric(X_train['int_rate'])
X_test['int_rate'] = pd.to_numeric(X_test['int_rate'])

# Remove dots in the revol_util feature
X_train['revol_util'] = X_train.revol_util.str.replace(',', '.')
X_test['revol_util'] = X_test.revol_util.str.replace(',', '.')

# Remove percent signs in the revol_util feature
X_train['revol_util'] = X_train.revol_util.str.replace('%', '')
X_test['revol_util'] = X_test.revol_util.str.replace('%', '')

# Convert the revol_util feature to numeric
X_train['revol_util'] = pd.to_numeric(X_train['revol_util'])
X_test['revol_util'] = pd.to_numeric(X_test['revol_util'])

# Convert the total_acc feature to numeric
X_train['total_acc'] = pd.to_numeric(X_train['total_acc'])
X_test['total_acc'] = pd.to_numeric(X_test['total_acc'])

# Convert the total_rec_int feature to numeric
X_train['total_rec_int'] = pd.to_numeric(X_train['total_rec_int'])
X_test['total_rec_int'] = pd.to_numeric(X_test['total_rec_int'])

# Remove commas and convert the funded_amnt_inv feature to numeric
X_train['funded_amnt_inv'] = X_train['funded_amnt_inv'].str.replace(',', '.')
X_test['funded_amnt_inv'] = X_test['funded_amnt_inv'].str.replace(',', '.')
X_train['funded_amnt_inv'] = pd.to_numeric(X_train['funded_amnt_inv'])
X_test['funded_amnt_inv'] = pd.to_numeric(X_test['funded_amnt_inv'])

# Convert the total_pymnt_inv feature to numeric
X_train['total_pymnt_inv'] = pd.to_numeric(X_train['total_pymnt_inv'])
X_test['total_pymnt_inv'] = pd.to_numeric(X_test['total_pymnt_inv'])

# Convert the delinq_2yrs feature to numeric
X_train['delinq_2yrs'] = pd.to_numeric(X_train['delinq_2yrs'])
X_test['delinq_2yrs'] = pd.to_numeric(X_test['delinq_2yrs'])

# Convert the dti feature to numeric
X_train['dti'] = pd.to_numeric(X_train['dti'])
X_test['dti'] = pd.to_numeric(X_test['dti'])

# Convert the total_pymnt feature to numeric
X_train['total_pymnt'] = pd.to_numeric(X_train['total_pymnt'])
X_test['total_pymnt'] = pd.to_numeric(X_test['total_pymnt'])

# Convert the collections_12_mths_ex_med feature to numeric
X_train['collections_12_mths_ex_med'] = pd.to_numeric(X_train['collections_12_mths_ex_med'])
X_test['collections_12_mths_ex_med'] = pd.to_numeric(X_test['collections_12_mths_ex_med'])

# Feature Selection

# Split the training set into two parts: categorical and numerical
X_train_cat = X_train.select_dtypes(include='object').copy()
X_train_num = X_train.select_dtypes(include='number').copy()

# Create an empty dictionary to store chi-squared results
chi2_check = {}

# Iterate through each column in the training set to calculate the chi-statistic with the target variable
for column in X_train_cat:
    chi, p, dof, ex = chi2_contingency(pd.crosstab(y_train, X_train_cat[column]))
    chi2_check.setdefault('Feature', []).append(column)
    chi2_check.setdefault('p-value', []).append(round(p, 10))

# Convert the dictionary to a DataFrame
chi2_result = pd.DataFrame(data=chi2_check)
chi2_result.sort_values(by=['p-value'], ascending=True, ignore_index=True, inplace=True)
chi2_result

# Fill missing values in X_train_num with their mean values
X_train_num.fillna(X_train_num.mean(), inplace=True)

# Calculate F Statistic and p values
F_statistic, p_values = f_classif(X_train_num, y_train)

# Convert to a DataFrame
ANOVA_F_table = pd.DataFrame(data={'Numerical_Feature': X_train_num.columns.values, 'F-Score': F_statistic, 'p values': p_values.round(decimals=10)})
ANOVA_F_table.sort_values(by=['F-Score'], ascending=False, ignore_index=True, inplace=True)
ANOVA_F_table

# To simplify, we will only keep the top 10 numerical features and calculate pair-wise correlations among them.

# Store the top 10 numerical features in a list
top_num_features = ANOVA_F_table.iloc[:10, 0].to_list()

# Calculate pair-wise correlations among them
corrmat = X_train_num[top_num_features].corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corrmat)

# Features open_acc and total_pymnt_inv can be dropped due to multicollinearity with other features.

# Create a list of columns to be dropped
drop_columns_list = ANOVA_F_table.iloc[10:, 0].to_list()
drop_columns_list.extend(chi2_result.iloc[3:, 0].to_list())
drop_columns_list.extend(['open_acc', 'total_pymnt_inv'])

# Function to drop columns from a DataFrame
def drop_columns(df, columns_list):
    df.drop(columns=columns_list, inplace=True)

# Use the function to drop columns from X_train
drop_columns(X_train, drop_columns_list)

# Check X_train info
X_train.info()

# Function to create dummy variables
def create_dummies(df, columns_list):
    df_dummies = []
    for col in columns_list:
        df_dummies.append(pd.get_dummies(df[col], prefix=col, prefix_sep=':'))
    df_dummies = pd.concat(df_dummies, axis=1)
    df = pd.concat([df, df_dummies], axis=1)
    return df

# Apply the function to the remaining categorical features
X_train = create_dummies(X_train, ['home_ownership', 'verification_status', 'addr_state'])

# Drop the assigned columns from the test set
drop_columns(X_test, drop_columns_list)

# Create dummy variables in X_test
X_test = create_dummies(X_test, ['home_ownership', 'verification_status', 'addr_state'])

# Re-index the dummy variables in the test set to ensure all column features in the training set are also present in the test set
X_test = X_test.reindex(labels=X_train.columns, axis=1, fill_value=0)

# WoE Binning/Feature Engineering

# Create copies of the dataset for WoE preprocessing
X_train_prepr = X_train.copy()
y_train_prepr = y_train.copy()
X_test_prepr = X_test.copy()
y_test_prepr = y_test.copy()

# Function to analyze WoE for discrete types
def woe_discrete(df, cat_variable_name, y_df):
    df = pd.concat([df[cat_variable_name], y_df], axis=1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop=True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

# Set the default style of plots to seaborn style
sns.set()

# Function to plot WoE
def plot_by_woe(df_WoE, rotation_of_x_axis_labels=0):
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize=(18, 6))
    plt.plot(x, y, marker='o', linestyle='--', color='k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title('Weight of Evidence by ' + df_WoE.columns[0])
    plt.xticks(rotation=rotation_of_x_axis_labels)

# Create a temporary dataframe to store discrete WoE for the 'home_ownership' feature
df_temp = woe_discrete(X_train_prepr, 'home_ownership', y_train_prepr)

# Plot the 'home_ownership' feature
plot_by_woe(df_temp)

# Create a temporary dataframe to store discrete WoE for the 'verification_status' feature
df_temp = woe_discrete(X_train_prepr, 'verification_status', y_train_prepr)

# Plot the 'verification_status' feature
plot_by_woe(df_temp)

# Create a temporary dataframe to store discrete WoE for the 'addr_state' feature
df_temp = woe_discrete(X_train_prepr, 'addr_state', y_train_prepr)

# Plot the 'addr_state' feature
plot_by_woe(df_temp, 90)

# Function to analyze ordered WoE for continuous types
def woe_ordered_continuous(df, continuous_variable_name, y_df):
    df = pd.concat([df[continuous_variable_name], y_df], axis=1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

# Create a temporary dataframe to store ordered WoE for the 'loan_amnt' feature
df_temp = woe_ordered_continuous(X_train_prepr, 'loan_amnt', y_train_prepr)

# Create a temporary dataframe to store ordered WoE for the 'term' feature
df_temp = woe_ordered_continuous(X_train_prepr, 'term', y_train_prepr)

# Plot the 'term' feature
plot_by_woe(df_temp)

# Fine-classing using the 'cut' method
X_train_prepr['int_rate'] = pd.cut(X_train_prepr['int_rate'], 50)

# Create a temporary dataframe to store ordered WoE for the 'int_rate' feature
df_temp = woe_ordered_continuous(X_train_prepr, 'int_rate', y_train_prepr)

# Plot the 'int_rate' feature
plot_by_woe(df_temp, 90)

# Fine-classing using the 'cut' method
X_train_prepr['annual_inc'] = pd.cut(X_train_prepr['annual_inc'], 50)

# Create a temporary dataframe to store ordered WoE for the 'annual_inc' feature
df_temp = woe_ordered_continuous(X_train_prepr, 'annual_inc', y_train_prepr)

# Plot the 'annual_inc' feature
plot_by_woe(df_temp, 90)

# Fine-classing using the 'cut' method
X_train_prepr['dti'] = pd.cut(X_train_prepr['dti'], 50)

# Create a temporary dataframe to store ordered WoE for the 'dti' feature
df_temp = woe_ordered_continuous(X_train_prepr, 'dti', y_train_prepr)

# Plot the 'dti' feature
plot_by_woe(df_temp, 90)

# Fine-classing using the 'cut' method for 'revol_util' feature
X_train_prepr['revol_util'] = pd.cut(X_train_prepr['revol_util'], 50)

# Create a temporary dataframe to store ordered WoE for the 'revol_util' feature
df_temp = woe_ordered_continuous(X_train_prepr, 'revol_util', y_train_prepr)

# Plot the 'revol_util' feature
plot_by_woe(df_temp, 90)

# Filter out observations with 'total_acc' > 50
X_train_prepr_temp = X_train_prepr[X_train_prepr['total_acc'] <= 50].copy()

# Fine-classing using the 'cut' method for 'total_acc' feature
X_train_prepr_temp['total_acc_factor'] = pd.cut(X_train_prepr_temp['total_acc'], 20)

# Create a temporary dataframe to store ordered WoE for the 'total_acc' feature
df_temp = woe_ordered_continuous(X_train_prepr_temp, 'total_acc_factor', y_train_prepr[X_train_prepr_temp.index])

# Plot the 'total_acc' feature
plot_by_woe(df_temp, 90)

# Fine-classing using the 'cut' method for 'total_pymnt' feature
X_train_prepr_temp['total_pymnt_factor'] = pd.cut(X_train_prepr_temp['total_pymnt'], 7)

# Create a temporary dataframe to store ordered WoE for the 'total_pymnt' feature
df_temp = woe_ordered_continuous(X_train_prepr_temp, 'total_pymnt_factor', y_train_prepr[X_train_prepr_temp.index])

# Plot the 'total_pymnt' feature
plot_by_woe(df_temp, 90)

# Define a list of reference categories for each feature
ref_categories = ['total_pymnt:>25,000', 'revol_util:>1.0', 'dti:>35.191', 
                  'annual_inc:>150K', 'int_rate:>20.281', 'term:60', 
                  'verification_status:Not Verified', 
                  'home_ownership:MORTGAGE', 'addr_state:WY']


# Create a class for creating dummy categorical features
class WoE_Binning(BaseEstimator, TransformerMixin):
    def __init__(self, X):
        self.X = X
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.loc[:, 'addr_state:AK': 'addr_state:WY']
        X_new['home_ownership:OWN'] = X.loc[:, 'home_ownership:OWN']
        X_new['home_ownership:MORTGAGE'] = X.loc[:, 'home_ownership:MORTGAGE']
        X_new['home_ownership:OTHER_NONE_RENT'] = sum([X['home_ownership:OTHER'], X['home_ownership:RENT']])
        X_new = pd.concat([X_new, X.loc[:, 'verification_status:Not Verified':'verification_status:Verified']], axis=1)
        X_new['term:36'] = np.where((X['term'] == 36), 1, 0)
        X_new['term:60'] = np.where((X['term'] == 60), 1, 0)
        X_new['int_rate:<7.071'] = np.where((X['int_rate'] <= 7.071), 1, 0)
        X_new['int_rate:7.071-10.374'] = np.where((X['int_rate'] > 7.071) & (X['int_rate'] <= 10.374), 1, 0)
        X_new['int_rate:10.374-13.676'] = np.where((X['int_rate'] > 10.374) & (X['int_rate'] <= 13.676), 1, 0)
        X_new['int_rate:13.676-15.74'] = np.where((X['int_rate'] > 13.676) & (X['int_rate'] <= 15.74), 1, 0)
        X_new['int_rate:15.74-20.281'] = np.where((X['int_rate'] > 15.74) & (X['int_rate'] <= 20.281), 1, 0)
        X_new['int_rate:>20.281'] = np.where((X['int_rate'] > 20.281), 1, 0)
        X_new['annual_inc:missing'] = np.where(X['annual_inc'].isnull(), 1, 0)
        X_new['annual_inc:<28,555'] = np.where((X['annual_inc'] <= 28555), 1, 0)
        X_new['annual_inc:28,555-37,440'] = np.where((X['annual_inc'] > 28555) & (X['annual_inc'] <= 37440), 1, 0)
        X_new['annual_inc:37,440-61,137'] = np.where((X['annual_inc'] > 37440) & (X['annual_inc'] <= 61137), 1, 0)
        X_new['annual_inc:61,137-81,872'] = np.where((X['annual_inc'] > 61137) & (X['annual_inc'] <= 81872), 1, 0)
        X_new['annual_inc:81,872-102,606'] = np.where((X['annual_inc'] > 81872) & (X['annual_inc'] <= 102606), 1, 0)
        X_new['annual_inc:102,606-120,379'] = np.where((X['annual_inc'] > 102606) & (X['annual_inc'] <= 120379), 1, 0)
        X_new['annual_inc:120,379-150,000'] = np.where((X['annual_inc'] > 120379) & (X['annual_inc'] <= 150000), 1, 0)
        X_new['annual_inc:>150K'] = np.where((X['annual_inc'] > 150000), 1, 0)
        X_new['dti:<=1.6'] = np.where((X['dti'] <= 1.6), 1, 0)
        X_new['dti:1.6-5.599'] = np.where((X['dti'] > 1.6) & (X['dti'] <= 5.599), 1, 0)
        X_new['dti:5.599-10.397'] = np.where((X['dti'] > 5.599) & (X['dti'] <= 10.397), 1, 0)
        X_new['dti:10.397-15.196'] = np.where((X['dti'] > 10.397) & (X['dti'] <= 15.196), 1, 0)
        X_new['dti:15.196-19.195'] = np.where((X['dti'] > 15.196) & (X['dti'] <= 19.195), 1, 0)
        X_new['dti:19.195-24.794'] = np.where((X['dti'] > 19.195) & (X['dti'] <= 24.794), 1, 0)
        X_new['dti:24.794-35.191'] = np.where((X['dti'] > 24.794) & (X['dti'] <= 35.191), 1, 0)
        X_new['dti:>35.191'] = np.where((X['dti'] > 35.191), 1, 0) 
        X_new['revol_util:missing'] = np.where(X['revol_util'].isnull(), 1, 0)
        X_new['revol_util:<0.1'] = np.where((X['revol_util'] <= 0.1), 1, 0)
        X_new['revol_util:0.1-0.2'] = np.where((X['revol_util'] > 0.1) & (X['revol_util'] <= 0.2), 1, 0)
        X_new['revol_util:0.2-0.3'] = np.where((X['revol_util'] > 0.2) & (X['revol_util'] <= 0.3), 1, 0)
        X_new['revol_util:0.3-0.4'] = np.where((X['revol_util'] > 0.3) & (X['revol_util'] <= 0.4), 1, 0)
        X_new['revol_util:0.4-0.5'] = np.where((X['revol_util'] > 0.4) & (X['revol_util'] <= 0.5), 1, 0)
        X_new['revol_util:0.5-0.6'] = np.where((X['revol_util'] > 0.5) & (X['revol_util'] <= 0.6), 1, 0)
        X_new['revol_util:0.6-0.7'] = np.where((X['revol_util'] > 0.6) & (X['revol_util'] <= 0.7), 1, 0)
        X_new['revol_util:0.7-0.8'] = np.where((X['revol_util'] > 0.7) & (X['revol_util'] <= 0.8), 1, 0)
        X_new['revol_util:0.8-0.9'] = np.where((X['revol_util'] > 0.8) & (X['revol_util'] <= 0.9), 1, 0)
        X_new['revol_util:0.9-1.0'] = np.where((X['revol_util'] > 0.9) & (X['revol_util'] <= 1.0), 1, 0)
        X_new['revol_util:>1.0'] = np.where((X['revol_util'] > 1.0), 1, 0)
        X_new['total_pymnt:<10,000'] = np.where((X['total_pymnt'] <= 10000), 1, 0)
        X_new['total_pymnt:10,000-15,000'] = np.where((X['total_pymnt'] > 10000) & (X['total_pymnt'] <= 15000), 1, 0)
        X_new['total_pymnt:15,000-20,000'] = np.where((X['total_pymnt'] > 15000) & (X['total_pymnt'] <= 20000), 1, 0)
        X_new['total_pymnt:20,000-25,000'] = np.where((X['total_pymnt'] > 20000) & (X['total_pymnt'] <= 25000), 1, 0)
        X_new['total_pymnt:>25,000'] = np.where((X['total_pymnt'] > 25000), 1, 0)
        X_new.drop(columns=ref_categories, inplace=True)
        return X_new

# Define a pipeline for modeling
reg = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.15)
woe_transform = WoE_Binning(X)
pipeline = Pipeline(steps=[('woe', woe_transform), ('model', reg)])

# Define cross-validation criteria using RepeatedStratifiedKFold to handle class imbalance
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# Fit and evaluate the logistic regression pipeline using the above cross-validation
scores = cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=cv)
AUROC = np.mean(scores)
GINI = AUROC * 2 - 1

# Display mean AUROC and Gini scores
print('Mean AUROC: %.4f' % (AUROC))
print('Gini: %.4f' % (GINI))

# Fit the pipeline with the entire training set
pipeline.fit(X_train, y_train)

# Transform the training set using WoE_Binning class
X_train_woe_transformed = woe_transform.fit_transform(X_train)

# Store the column names as a list
feature_name = X_train_woe_transformed.columns.values

# Create a summary table for logistic regression model
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)

# Create a new column in the dataframe called Coefficients with values from the transpose of coefficients from the logistic regression model
summary_table['Coefficients'] = np.transpose(pipeline['model'].coef_)

# Add an index to each row to store the intercept of the model in the first row
summary_table.index = summary_table.index + 1

# Assign the intercept to the first row
summary_table.loc[0] = ['Intercept', pipeline['model'].intercept_[0]]

# Sort the dataframe based on index
summary_table.sort_index(inplace=True)
summary_table

# Make predictions on the test set
y_hat_test = pipeline.predict(X_test)

# Get prediction probabilities
y_hat_test_proba = pipeline.predict_proba(X_test)

# Select probabilities of the positive class only
y_hat_test_proba = y_hat_test_proba[:, 1]

# Create a new dataframe with the actual class and prediction probabilities
y_test_temp = y_test.copy()
y_test_temp.reset_index(drop=True, inplace=True)
y_test_proba = pd.concat([y_test_temp, pd.DataFrame(y_hat_test_proba)], axis=1)

# Check the shape to ensure the number of rows is the same as in y_test
y_test_proba.shape

# Rename columns
y_test_proba.columns = ['y_test_class_actual', 'y_hat_test_proba']

# Align the index of the dataframes
y_test_proba.index = X_test.index
y_test_proba.head()

# Assign a threshold value to distinguish between good and bad
tr = 0.5

# Create a new column for predicted class based on prediction probabilities and threshold value
y_test_proba['y_test_class_predicted'] = np.where(y_test_proba['y_hat_test_proba'] > tr, 1, 0)

# Create a confusion matrix
confusion_matrix(y_test_proba['y_test_class_actual'], y_test_proba['y_test_class_predicted'], normalize='all')

# Get values required for plotting the ROC curve
fpr, tpr, thresholds = roc_curve(y_test_proba['y_test_class_actual'], y_test_proba['y_hat_test_proba'])

# Plot the ROC curve
plt.plot(fpr, tpr)

# Plot a diagonal line to represent the no-skill classifier
plt.plot(fpr, fpr, linestyle='--', color='k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')

# Calculate the area under the ROC curve (AUROC) on the test set
AUROC = roc_auc_score(y_test_proba['y_test_class_actual'], y_test_proba['y_hat_test_proba'])
AUROC

# Calculate Gini from AUROC
Gini = AUROC * 2 - 1
Gini

# Plot the PR curve
no_skill = len(y_test[y_test == 1]) / len(y)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

# Calculate precision-recall curve inputs
precision, recall, thresholds = precision_recall_curve(y_test_proba['y_test_class_actual'], y_test_proba['y_hat_test_proba'])

# Plot PR curve
plt.plot(recall, precision, marker='.', label='Logistic')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision-Recall curve')

# Calculate the area under the PR curve (AUC_PR)
auc_pr = auc(recall, precision)
auc_pr

# Create a new DataFrame for reference categories
df_ref_categories = pd.DataFrame(ref_categories, columns=['Feature name'])

# Add a second column 'Coefficients' with values of 0
df_ref_categories['Coefficients'] = 0

# Concatenate the two dataframes
df_scorecard = pd.concat([summary_table, df_ref_categories])

# Reset the dataframe's index
df_scorecard.reset_index(inplace=True)

# Create a new column 'Original feature name' with values from the 'Feature name' column
df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]

# Define min and max threshold values for the scorecard
min_score = 300
max_score = 850

# Calculate the sum of minimum coefficients for each category of 'Original feature name'
min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()

# Calculate the sum of maximum coefficients for each category of 'Original feature name'
max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()

# Create a new column 'Score - Calculation' based on the coefficient ratio
df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)

# Update the calculation score for the Intercept
df_scorecard.loc[0, 'Score - Calculation'] = ((df_scorecard.loc[0, 'Coefficients'] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score

# Round the 'Score - Calculation' to the nearest integer and save it in a new column
df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()
df_scorecard

# Check the possible min and max score from the scorecard
min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].min().sum()
max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].max().sum()
print(min_sum_score_prel)
print(max_sum_score_prel)

# Evaluate based on estimated differences from Original feature name
df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']

# It seems we can obtain the Final Score by subtracting 1 from the Intercept
df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']
df_scorecard.loc[0, 'Score - Final'] = 580

# Check the possible min and max score from the scorecard again
print(df_scorecard.groupby('Original feature name')['Score - Final'].min().sum())
print(df_scorecard.groupby('Original feature name')['Score - Final'].max().sum())

# Calculate Credit Score on the Test set
X_test_woe_transformed = woe_transform.fit_transform(X_test)
X_test_woe_transformed.insert(0, 'Intercept', 1)

# Get the list of final scores from the scorecard
scorecard_scores = df_scorecard['Score - Final']

# Perform matrix dot product for the test set and scorecard scores
y_scores = X_test_woe_transformed.dot(scorecard_scores.values.reshape(98, 1))

# Convert to a DataFrame for further analysis
df_y_scores = pd.DataFrame(y_scores)

# Set Loan Approval Time Limit
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold: %f' % (best_thresh))

# Update the threshold value
tr = best_thresh

# Create a new column for predicted class based on prediction probability and threshold
y_test_proba['y_test_class_predicted'] = np.where(y_test_proba['y_hat_test_proba'] > tr, 1, 0)

# Create a confusion matrix
confusion_matrix(y_test_proba['y_test_class_actual'], y_test_proba['y_test_class_predicted'], normalize='all')

# Create a new dataframe consisting of ROC output thresholds
df_cutoffs = pd.DataFrame(thresholds, columns=['thresholds'])

# Calculate scores based on the threshold
df_cutoffs['Score'] = ((np.log(df_cutoffs['thresholds'] / (1 - df_cutoffs['thresholds'])) - min_sum_coef) * 
                       ((max_score - min_score) / (max_sum_coef - min_sum_coef)) + min_score).round()

# Create a function to assign a value of 1 when the prediction probability is greater than or equal to the threshold, and 0 otherwise, and then sum the column
def n_approved(p):
    return np.where(y_test_proba['y_hat_test_proba'] >= p, 1, 0).sum()

# Calculate the number of approved applications for each threshold
df_cutoffs['N Approved'] = df_cutoffs['thresholds'].apply(n_approved)

# Calculate the number of rejected applications for each threshold (total applications - approved applications)
df_cutoffs['N Rejected'] = y_test_proba['y_hat_test_proba'].shape[0] - df_cutoffs['N Approved']

# Approval rate is the ratio of approved applications to all applications
df_cutoffs['Approval Rate'] = df_cutoffs['N Approved'] / y_test_proba['y_hat_test_proba'].shape[0]

# Rejection rate is 1 - approval rate
df_cutoffs['Rejection Rate'] = 1 - df_cutoffs['Approval Rate']

# View approval and rejection rates for the ideal threshold
df_cutoffs[df_cutoffs['thresholds'].between(0.40166, 0.40167)]

# Compare rates with the default case, where the threshold is 0.5
df_cutoffs[df_cutoffs['thresholds'].between(0.5, 0.5001)]

# Display a scatter plot of the dataset
sns.scatterplot(data=dataset)

# Since the scatter plot shows no clear structure, we will reduce the dataset to 2 features using PCA for clustering
# Select only numerical columns from the dataset
datasetNumber = dataset.select_dtypes(include=np.number)

# Perform PCA with 2 components
pca = PCA(n_components=2)

# Transform the dataset using PCA
skl_pca = pca.fit_transform(datasetNumber)

# Convert the PCA result to a DataFrame
skl_pca_df = pd.DataFrame(data=skl_pca)

# Create a scatterplot of the PCA result
sns.scatterplot(x=0, y=1, data=skl_pca_df)



# Check the appropriate number of clusters to choose
for i, k in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10]):
    
    # Run the KMeans algorithm
    km = KMeans(n_clusters=k)
    y_predict = km.fit_predict(skl_pca_df)
    centroids = km.cluster_centers_

    # Get silhouette values
    silhouette_vals = silhouette_samples(skl_pca_df, y_predict)

    # Display silhouette score
    print("For clusters", k, "the silhouette score is: ", silhouette_score(skl_pca_df, y_predict))

# Based on the results, the highest silhouette score is obtained when there are 2 clusters.
# So, for further data analysis and interpretation, we will use KMeans with 2 clusters.

# Create KMeans with 2 clusters, as chosen above
km = KMeans(n_clusters=2, max_iter=100)
km.fit(skl_pca_df)

# Predict skl_pca_df with the defined KMeans
y_predict = km.fit_predict(skl_pca_df)

# Plot the clustering visualization
fig, ax = plt.subplots(figsize=(10, 10))
plt.scatter(skl_pca[y_predict == 0, 0], skl_pca[y_predict == 0, 1],
            c='green', label='Cluster 1')
plt.scatter(skl_pca[y_predict == 1, 0], skl_pca[y_predict == 1, 1],
            c='blue', label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            marker='*', s=300, c='red', label='Centroid')

# Add labels to the clustering visualization
plt.title('Visualization of clustered data', fontweight='bold')
plt.legend()
ax.set_aspect('equal')

# Based on the clustered data visualization, we can observe characteristics such as:
# - Data tends to be "stacked".
# - The further down, the more data is "stacked".
# - Clusters 1 and 2 cannot be separated by a straight line (non-linearly separable) due to some overlapping data points.
# - Centroids are precisely at the center of the data clusters.