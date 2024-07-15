import streamlit as st  ##importing the necessary libraries
import pandas as pd 
import numpy as np 
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier

st.write(
 """
 # Penguin Classification App


This app tells the various different types of ***Penguins***!
"""


)

st.sidebar.header('User Input Parameters')

def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()

penguins_raw = pd.read_csv("C:/Users/KIIT0001/Documents/kaggle_datasets/penguins_cleaned.csv")
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

encode=['sex','island']
for col in encode:
    dummy=pd.get_dummies(df[col],prefix=col)
    df=pd.concat([df,dummy],axis=1)
    del df[col]
df=df[:1]

st.subheader("User Input Features")
st.write(df)

load_clf=pkl.load(open('penguin_classification.pkl','rb'))

prediction=load_clf.predict(df)
prediction_proba=load_clf.predict_proba(df)

st.subheader('Prediction')
penguin_species=np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguin_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)