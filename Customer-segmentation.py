import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title('Customer Segmentation')

# Step 1: Upload the CSV File
st.subheader('Upload CSV File')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data=pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview:")
    st.write(data.head(5))
    
    data.dropna(axis=0, inplace=True)  # Drop rows with missing values
    data.drop_duplicates(inplace=True)  
    
    X = data.copy()
    if 'CustomerID' in X.columns:
        X = X.drop('CustomerID', axis=1)

    # Encode Gender if exists
    if 'Gender' in X.columns:
        X['Gender'] = X['Gender'].map({'Male':0, 'Female':1})
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.sidebar.subheader("Clustering Options")
    k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5)

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    data['Cluster'] = clusters
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    data['PCA1'] = X_pca[:,0]
    data['PCA2'] = X_pca[:,1]

    st.subheader("Customer Segments (PCA Projection)")
    plt.figure(figsize=(10,6))
    palette = sns.color_palette("Set2", n_colors=k)
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette=palette, s=120, alpha=0.8)
    plt.title("Customer Segmentation (PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    st.pyplot(plt.gcf())
    plt.clf()

    numeric_cols = data.select_dtypes(include='number').columns.drop(['Cluster', 'PCA1', 'PCA2'])
    cluster_summary = data.groupby('Cluster')[numeric_cols].mean()
    cluster_summary['Count'] = data['Cluster'].value_counts().sort_index()
    st.subheader("Cluster Summary")
    st.dataframe(cluster_summary.round(2))
 
    st.subheader("Key Insights")
    st.markdown("- **High-Value Customers**: Identify clusters with high 'Annual Income' & 'Spending Score'.")
    st.markdown("- **Target Low-Spending Segments**: Focus marketing campaigns on clusters with low 'Spending Score'.")
    st.markdown("- **Customer Demographics**: Use age/gender breakdowns to tailor promotions.")

    # Optional: Visualize cluster averages
    st.subheader("Cluster Feature Comparison")
    st.bar_chart(cluster_summary.drop(columns=['Count']))

    
else:
    st.info("Upload a CSV file to get started.")

