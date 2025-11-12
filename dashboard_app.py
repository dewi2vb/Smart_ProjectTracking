import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Smart Surveillance Dashboard", layout="wide")

st.title("📊 Dashboard Pelacakan Individu Kampus")
st.markdown("Visualisasi hasil deteksi dan pelacakan berbasis YOLOv8 + Deep SORT")

# Load data
df = pd.read_csv('movement_log.csv')

# Statistik ringkas
col1, col2, col3 = st.columns(3)
col1.metric("Total Frame", df['frame'].max())
col2.metric("Jumlah Individu", df['track_id'].nunique())
col3.metric("Total Data Log", len(df))

# Heatmap
st.subheader("Heatmap Pergerakan Individu")
x = df['x_center'].values
y = df['y_center'].values
heatmap, _, _ = np.histogram2d(x, y, bins=(50, 50))
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
st.pyplot(plt)

# Log data
st.subheader("Log Pelacakan")
st.dataframe(df)
