import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame and the target variable is 'G3'

df = pd.read_csv("stud.csv")

# Define your features (X) and target (y)
X = df.drop(columns=['G3'])
y = df['G3']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a KNN model
knn_model = KNeighborsClassifier(n_neighbors=8, metric='manhattan')
knn_model.fit(X_train, y_train)

# Compute SHAP values
explainer = shap.Explainer(knn_model, X_train)
shap_values = explainer(X_test)

# Calculate mean absolute SHAP values
shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)
mean_abs_shap_values = shap_values_df.abs().mean().sort_values(ascending=False)

# Get top 10 features
top_10_features = mean_abs_shap_values.head(10)

# Create two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.write("Mean Absolute SHAP Values (Feature Importance):")
    st.dataframe(mean_abs_shap_values)

    # Plot feature importance
    fig, ax = plt.subplots()
    sns.barplot(x=top_10_features.values, y=top_10_features.index, palette='viridis', ax=ax)
    ax.set_title('Top 10 Feature Importances by Mean Absolute SHAP Value')
    ax.set_xlabel('Mean Absolute SHAP Value')
    ax.set_ylabel('Features')
    st.pyplot(fig)
    
with col2:
    st.write("Feature Description:")
    feature_description = df.describe().T
    st.dataframe(feature_description)
