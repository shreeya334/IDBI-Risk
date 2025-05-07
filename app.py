import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set page config
st.set_page_config(page_title="Credit Risk Analysis", layout="wide")

st.title("Credit Risk Analysis using Machine Learning")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("IDBI_credit_data.csv")
    return df

df = load_data()

# Display initial data
st.subheader("Raw Dataset (First 4 Rows)")
st.write(df.head(4))

# Assign risk
def assign_risk(row):
    if row['Credit amount'] > 5000 or row['Duration'] > 36:
        return 'bad'
    else:
        return 'good'

df['Risk'] = df.apply(assign_risk, axis=1)

# Drop missing values
df.dropna(inplace=True)

st.subheader("Dataset After Assigning Risk and Cleaning")
st.write(df.head(4))
st.write("**Missing Values:**", df.isnull().sum().sum())

# Data Description
st.subheader("Data Description")
st.write(df.describe(include='all'))

# Box Plot
st.subheader("Box Plot for Numerical Columns")
fig1, ax1 = plt.subplots(figsize=(6,4))
df[['Age', 'Credit amount', 'Duration']].boxplot(ax=ax1)
st.pyplot(fig1)

# Histograms
st.subheader("Histograms of Numerical Data")
fig2 = df.hist(bins=30, figsize=(10, 6), color='r')
st.pyplot(plt.gcf())

# Categorical bar plots
st.subheader("Categorical Data Distribution")
categorical_columns = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']
for column in categorical_columns:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=df[column].value_counts().index, y=df[column].value_counts().values, ax=ax)
    ax.set_title('Distribution of ' + column)
    st.pyplot(fig)

# Correlation matrix
st.subheader("Correlation Matrix")
fig_corr, ax_corr = plt.subplots()
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, ax=ax_corr)
st.pyplot(fig_corr)

# Pairplot
st.subheader("Pairplot of Dataset")
st.pyplot(sns.pairplot(df))

# Relationship with target
st.subheader("Relationship with Risk")
for column in df.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    if df[column].dtype != 'object':
        sns.boxplot(x='Risk', y=column, data=df, ax=ax)
    else:
        sns.countplot(x=column, hue='Risk', data=df, ax=ax)
    ax.set_title(f'Relationship of Risk with {column}')
    st.pyplot(fig)

# Preprocessing
X = df.drop('Risk', axis=1)
y = df['Risk']

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)])

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
X_test = preprocessor.transform(X_test)

# Model training and evaluation
st.subheader("Model Evaluation")

accuracy_results = {}

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_valid_preds = knn.predict(X_valid)
accuracy_results['KNN (Validation)'] = accuracy_score(y_valid, y_valid_preds)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_valid_preds = logreg.predict(X_valid)
accuracy_results['LogReg (Validation)'] = accuracy_score(y_valid, y_valid_preds)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_valid_preds = rf.predict(X_valid)
accuracy_results['RF (Validation)'] = accuracy_score(y_valid, y_valid_preds)

accuracy_table = pd.DataFrame(list(accuracy_results.items()), columns=['Model', 'Accuracy'])
st.dataframe(accuracy_table)

# Optional: Highlight best model
best_model = accuracy_table.loc[accuracy_table['Accuracy'].idxmax()]
st.success(f"Best Model: **{best_model['Model']}** with Accuracy: **{best_model['Accuracy']:.2f}**")
