import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

st.set_page_config(page_title='Spam or Ham', page_icon="https://github.com/Strociak/Lamda/blob/main/StreamlitBuild/icon.png?raw=true")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>Spam or Ham?</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Spam or Ham description here.</h4>", unsafe_allow_html=True)

emaildataset = pd.read_csv('OurDataset.csv')
subjectdataset = pd.read_csv('OurSubjectDataset.csv')

z = emaildataset['email']
y = emaildataset['class']
z_train, z_test, y_train, y_test = train_test_split(z,y,test_size = 0.2)

vect = CountVectorizer(stop_words='english')
vect.fit(z_train.values)

z_train_df = vect.transform(z_train)
z_test_df = vect.transform(z_test)

model = MultinomialNB()
model.fit(z_train_df,y_train)

x = subjectdataset['subject']
w = subjectdataset['class']
x_train, x_test, w_train, w_test = train_test_split(x,w,test_size = 0.2)

vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(x_train.values)

x_train_df = vectorizer.transform(x_train)
x_test_df = vectorizer.transform(x_test)

subjectmodel = MultinomialNB()
subjectmodel.fit(x_train_df,w_train)

entryframe = st.empty()

#def user_input_features():
msg = entryframe.text_input('Please input any of the following from the email you wish to test: its sender, its subject, or its body text:', value="", key="input")
#    return entryframe
    
#msg = user_input_features()

def stringPredictEmail(entry):
    try:
        msgarray = (vect.transform([entry]))
        resultbool = model.predict(msgarray)
        if (resultbool):
            result = "The email is most likley spam."
        else:
            result = "The email is most likley ham."
    except ValueError:
        pass
    return result
    
def stringPredictSubject(entry):
    try:
        msgarray = (vectorizer.transform([entry]))
        resultbool = subjectmodel.predict(msgarray)
        if (resultbool):
            result = "The email is most likley spam."
        else:
            result = "The email is most likley ham."
    except ValueError:
        pass
    return result
    
def blacklistAddresses(entry):
    try:
        tempbool = False
        with open('OurAddressDataset.csv', 'rt') as f:
            reader = csv.reader(f, delimiter = ',')
            for row in reader:
                for field in row:
                    if field == entry:
                        tempbool = True
        if (tempbool):
            result = "The email is most likley spam."
        else:
            result = "The email is most likley ham."
    except ValueError:
        pass
    return result

col1, col2, col3 = st.columns(3)

with col1:
    senderbutton = st.button("Sender")
with col2:
    subjectbutton = st.button("Subject")
with col3:
    bodybutton = st.button("Body Text")
    
#clearbutton = st.button("Clear")
    
if(senderbutton):
    #st.write(msg)
    st.write(blacklistAddresses(msg))
    #msg = entryframe.text_input('Please input any of the following from the email you wish to test: its sender, its subject, or its body text:', value="", key="2")
    
if(subjectbutton):
   #st.write(msg)
   st.write(stringPredictSubject(msg))
   #msg = entryframe.text_input('Please input any of the following from the email you wish to test: its sender, its subject, or its body text:', value="", key="3")
    
if(bodybutton):
    #st.write(msg)
    st.write(stringPredictEmail(msg))
    #msg = entryframe.text_input('Please input any of the following from the email you wish to test: its sender, its subject, or its body text:', value="", key="4")

#if 'temp' not in st.session_state:
#    st.session_state.temp = ''
    
#def clear():
#    st.session_state.temp = st.session_state.input
#    st.session_state.input = ''

#if(clearbutton):
#    clear()
