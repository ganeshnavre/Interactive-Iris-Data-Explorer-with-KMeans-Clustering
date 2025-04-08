# Interactive-Iris-Data-Explorer-with-KMeans-Clustering
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans

# Title
st.title("")

# Load dataset
@st.cache_data
def load_data():
    return sns.load_dataset('iris')

df = load_data()

# Show Data
if st.checkbox("Show Raw Data"):
    st.write(df)

# Plot 1: Interactive Plotly Scatter
st.subheader("🌸 Interactive Scatter (Petal vs Sepal)")
fig1 = px.scatter(
    df, x='sepal_length', y='petal_length',
    color='species', size='sepal_width',
    hover_data=['petal_width']
)
st.plotly_chart(fig1)

# Clustering with KMeans
st.subheader("KMeans Clustering")
X = df[['sepal_length', 'petal_length']]
kmeans = KMeans(n_clusters=3, n_init='auto')
df['cluster'] = kmeans.fit_predict(X)

fig2 = px.scatter(
    df, x='sepal_length', y='petal_length',
    color=df['cluster'].astype(str),
    title='KMeans Clustered Sepal vs Petal Length'
)
st.plotly_chart(fig2)

# Heatmap
st.subheader("Correlation Heatmap")
corr = df.drop('cluster', axis=1).select_dtypes('number').corr()
st.dataframe(corr.style.background_gradient(cmap='coolwarm'))

# Footer
st.markdown("---")
st.markdown(".")

