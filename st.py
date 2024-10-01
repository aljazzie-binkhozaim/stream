import streamlit as st
import joblib
st.title('Hello world')

model = joblib.load('linear.pkl')

st.write('## Salary Prediction')

x = st.slider('Experience', 0,40)

y= model.predict([[x]]) # predict always accepts 2 dimensional

st.write('salary is : ', y)

