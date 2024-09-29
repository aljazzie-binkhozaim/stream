import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.title('**Hello** World!')

model = joblib.load('LinearRegression.pkl')

x = st.slider('Experience Years: ', 0, 20)
y = model.predict([[x]])

st.write(f'Result: {y}')

np.random.seed(42)
mean = [10, 20]
cov = [[5, 2], [2, 2]]  # covariance matrix
x, y = np.random.multivariate_normal(mean, cov, 500).T
plot = plt.scatter(x,y)
st.write(plt.gcf())

