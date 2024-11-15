import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
st.set_page_config(layout='wide')
df=pd.read_csv("./Loan approval prediction.csv")
st.subheader("Loading Data")

st.write(df.head())
# Show dataset information
st.write("### Dataset Information")
st.write(df.info())
st.dataframe(df.describe().T)
st.write("### EDA")

st.write("### Distribution of Loan Amounts")
if 'loan_amnt' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['loan_amnt'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Loan Amounts')
    ax.set_xlabel('Loan Amount')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
else:
    st.error("'loan_amnt' column not found in the dataset!")
    st.write("### Correlation Heatmap (Numeric Columns)")

# Check if there are numeric columns in the dataset
numeric_data = df.select_dtypes(include=['float64', 'int64'])  # Select numeric columns

if not numeric_data.empty:
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = numeric_data.corr()  # Compute correlation matrix
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)
else:
    st.warning("No numeric columns available in the dataset to compute a correlation heatmap.")
# Distribution of Age in Streamlit
st.write("### Distribution of Age")

# Check if the 'person_age' column exists in the dataset
if 'person_age' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['person_age'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
else:
    st.error("'person_age' column not found in the dataset!")
numeric_columns = [
    'person_age', 'person_income ', 'person_emp_length', 
    'loan_amnt', 'loan_int_rate', 
    'loan_percent_income', 'cb_person_cred_hist_length', 
    'loan_status'
]
# Strip whitespace from column names
df.columns = df.columns.str.strip()
print(df.columns.tolist())
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Detect outliers for each numeric column
outliers_iqr = {col: detect_outliers_iqr(df, col) for col in numeric_columns}

# Print the number of outliers detected in each column
for col, outliers in outliers_iqr.items():
    print(f'Outliers in {col}: {len(outliers)}')
sns.set(style="whitegrid")
custom_palette = sns.color_palette(["#81c784", "#388e3c", "#74c69d"])
# Boxplots of Numeric Columns in Streamlit
st.write("### Boxplots of Numeric Columns")

# Define the function for plotting boxplots
def plot_boxplots(df, columns):
    fig, axes = plt.subplots(nrows=(len(columns) + 2) // 3, ncols=3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.boxplot(y=df[col], ax=axes[i], color='skyblue')
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_ylabel('')
    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig

# Ensure there are numeric columns
if numeric_columns:
    fig = plot_boxplots(df, numeric_columns)
    st.pyplot(fig)
else:
    st.warning("No numeric columns available for boxplots.")

# Numeric Columns Distribution in Streamlit
st.write("### Distribution of Numeric Columns")

if numeric_columns:
    fig, axes = plt.subplots(nrows=(len(numeric_columns) // 3) + 1, ncols=3, figsize=(14, 12))
    axes = axes.flatten()
    custom_palette = sns.color_palette("Set2")  # Define a custom color palette

    for idx, col in enumerate(numeric_columns):
        sns.histplot(df[col], kde=True, ax=axes[idx], color=custom_palette[idx % 3], edgecolor="black", linewidth=1.2)
        axes[idx].set_title(f'{col} Distribution', fontsize=14, weight='bold', color="#388e3c")
        axes[idx].set_xlabel(col, fontsize=12, color="#555555")
        axes[idx].set_ylabel('Frequency', fontsize=12, color="#555555")
        sns.despine()

    # Hide unused subplots
    for idx in range(len(numeric_columns), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("No numeric columns available for plotting.")
# Categorical Columns Distribution in Streamlit
st.write("### Distribution of Categorical Columns")

categorical_columns1 = ['loan_intent']  # Update as needed

if categorical_columns1:
    for column in categorical_columns1:
        st.write(f"#### Distribution of {column}")
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.countplot(x=df[column], palette="Set2", ax=ax)
        ax.set_title(f'Distribution of {column}', fontsize=14)
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.warning("No categorical columns available for plotting.")
categorical_columns2 = ['cb_person_default_on_file', 'loan_grade', 'person_home_ownership']

# Title in the Streamlit app
st.write("## Categorical Columns Distribution")

# Check if the columns exist in the DataFrame
if all(col in df.columns for col in categorical_columns2):
    custom_palette = sns.color_palette("Set2")  # Custom palette
    fig, axes = plt.subplots(1, len(categorical_columns2), figsize=(15, 5))

    for i, column in enumerate(categorical_columns2):
        sns.countplot(x=df[column], palette=custom_palette, ax=axes[i])
        axes[i].set_title(f'Distribution of {column}', fontsize=12)
        axes[i].set_xlabel(column, fontsize=10)
        axes[i].set_ylabel('Count', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)  # Display the plot in Streamlit
else:
    missing_cols = [col for col in categorical_columns2 if col not in df.columns]
    st.error(f"The following columns are missing from the dataset: {missing_cols}")
categorical_columns=['loan_intent','cb_person_default_on_file', 'loan_grade','person_home_ownership']
# DATA CLEANING
df.drop(columns=['id'], inplace=True)
# List of numerical columns in your data
numerical_columns = ['person_income', 'loan_amnt', 'person_age', 'person_emp_length', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

# Z-Score Method for Outlier Detection
def detect_outliers_zscore(df, columns, threshold=3):
    outlier_indices = []
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        outlier_indices.extend(np.where(z_scores > threshold)[0])
    return set(outlier_indices)
# Detect outliers using Z-scores
outliers_zscore = detect_outliers_zscore(df, numerical_columns)
print(f'Number of outliers detected using Z-scores: {len(outliers_zscore)}')
# IQR Method for Outlier Detection
def detect_outliers_iqr(df, columns):
    outlier_indices = []
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices.extend(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index)
    return set(outlier_indices)
# Detect outliers using IQR
outliers_iqr = detect_outliers_iqr(df, numerical_columns)
print(f'Number of outliers detected using IQR: {len(outliers_iqr)}')
# Removing Outliers
# To remove the outliers from the DataFrame, combine the indices from both methods
outliers_combined = outliers_zscore.union(outliers_iqr)
df_no_outliers = df.drop(index=outliers_combined)
df_cleaned = df.drop(index=outliers_zscore)
print(f'Shape of DataFrame after outlier removal: {df_cleaned.shape}')

numerical_columns = ['person_income', 'loan_amnt', 'person_age', 'person_emp_length', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

# Check if numeric columns exist
if all(col in df_cleaned.columns for col in numerical_columns):
    correlation_matrix = df_cleaned[numerical_columns].corr()

    # Create a heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='Greens', linewidths=0.2, ax=ax)
    ax.set_title('Correlation Matrix of Numeric Features', fontsize=16)

    # Display the heatmap in Streamlit
    st.pyplot(fig)
else:
    missing_cols = [col for col in numerical_columns if col not in df_cleaned.columns]
    st.error(f"The following numeric columns are missing from the dataset: {missing_cols}")
st.write("## Loan Approval Based on Income and Credit History Length")

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    x='person_income', 
    y='cb_person_cred_hist_length', 
    hue='loan_status', 
    data=df_cleaned, 
    palette="Greens", 
    alpha=0.7,
    ax=ax
)
ax.set_title('Loan Approval Based on Income and Credit History Length', fontsize=16)
ax.set_xlabel('Annual Income')
ax.set_ylabel('Credit History Length')

# Display the plot in Streamlit
st.pyplot(fig)
st.write("## Pairplot: Income, Loan Amount, Credit History, and Loan Status")

selected_columns = ['person_income', 'loan_amnt', 'cb_person_cred_hist_length', 'loan_status']

# Generate Pairplot
with st.spinner("Generating pairplot..."):
    pairplot_fig = sns.pairplot(
        df_cleaned[selected_columns],
        hue='loan_status',
        palette=sns.color_palette("Greens"),
        diag_kind='kde'
    )
    pairplot_fig.fig.suptitle(
        'Pairplot of Income, Loan Amount, Credit History, and Loan Status', 
        y=1.02, 
        fontsize=16
    )

# Display the plot in Streamlit
st.pyplot(pairplot_fig)
warnings.filterwarnings('ignore')

# Define custom palette
custom_palette = sns.color_palette("Greens")

# Display histograms for numeric columns
st.write("## Distribution of Numeric Columns")

# Numeric Columns
numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns

# Create subplots for each numeric column
fig, axes = plt.subplots(
    nrows=(len(numeric_columns) // 3) + 1, ncols=3, figsize=(14, 12)
)
axes = axes.flatten()

for idx, col in enumerate(numeric_columns):
    sns.histplot(
        df_cleaned[col],
        kde=True,
        color=custom_palette[idx % len(custom_palette)],
        edgecolor="black",
        linewidth=1.2,
        ax=axes[idx]
    )
    axes[idx].set_title(f'{col} Distribution', fontsize=14, weight='bold', color="#388e3c")
    axes[idx].set_xlabel(col, fontsize=12, color="#555555")
    axes[idx].set_ylabel('Frequency', fontsize=12, color="#555555")
    sns.despine()

# Hide unused subplots
for ax in axes[len(numeric_columns):]:
    ax.axis('off')

plt.tight_layout()

# Render the plot in Streamlit
st.pyplot(fig)
# User selection for specific numeric columns
selected_columns = st.multiselect(
    "Select Numeric Columns to Compare with Loan Status:",
    options=numeric_columns,
    default=list(numeric_columns)
)

# Create boxplots only for the selected columns
if selected_columns:
    fig, axes = plt.subplots(nrows=len(selected_columns) // 3 + 1, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, column in enumerate(selected_columns):
        sns.boxplot(
            x='loan_status',
            y=column,
            data=df_cleaned,
            palette=custom_palette,
            ax=axes[idx]
        )
        axes[idx].set_title(f'{column} vs Loan Status', fontsize=14)
        axes[idx].set_xlabel('Loan Status', fontsize=12)
        axes[idx].set_ylabel(column, fontsize=12)

    for ax in axes[len(selected_columns):]:
        ax.axis('off')

    st.pyplot(fig)
st.title("Loan Status Prediction - Model Comparison")
st.write("This app allows you to evaluate various classification models for loan status prediction.")

# Upload Dataset
st.write("### Dataset Overview")
st.dataframe(df.head())

# Preprocessing
st.write("### Data Preprocessing")

# Encode Target Variable
label_encoder = LabelEncoder()
df['loan_status'] = label_encoder.fit_transform(df['loan_status'])

# Separate features and target
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numeric_columns = X.select_dtypes(include=['number']).columns

# One-Hot Encode Categorical Variables
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Train-Test Split
test_size = st.slider("Test Size (as percentage):", min_value=10, max_value=50, value=35, step=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=100, stratify=y)

# Scale Numeric Data
scaler = StandardScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

st.write("Data has been preprocessed successfully!")

# Model Evaluation
st.write("### Model Training and Evaluation")

# List of models
models = {
    "Logistic Regression": LogisticRegression(random_state=48, max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(random_state=48),
    "Decision Tree": DecisionTreeClassifier(random_state=100),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=48),
    "XGBoost": XGBClassifier(),
    "Naive Bayes": GaussianNB()
}

# Evaluate Models
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

# Display Results
st.write("### Model Performance")
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
st.dataframe(results_df)

# Select Model for Detailed Report
selected_model = st.selectbox("Select a model for detailed evaluation:", options=list(models.keys()))
if selected_model:
    model = models[selected_model]
    y_pred = model.predict(X_test)
    
    st.write(f"### Classification Report - {selected_model}")
    st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_.astype(str)))
    
    st.write(f"### Confusion Matrix - {selected_model}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))