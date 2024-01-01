import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

model = pickle.load(open('knn-paru-paru.sav', 'rb'))

st.title('Estimasi Pasien Yang Menderita Kanker Paru-Paru')


AGE = st.text_input('Usia Anda')
SMOKING = st.selectbox('Apakah pasien merokok?', ['YES', 'NO'])

if SMOKING == 'YES':
    SMOKING = 2
elif SMOKING == 'NO':
    SMOKING = 1
else:
    SMOKING = 0

YELLOW_FINGERS = st.selectbox('Apakah pasien jari pasien kuning?', ['YES', 'NO'])
    
if YELLOW_FINGERS == 'YES':
    YELLOW_FINGERS = 2
elif YELLOW_FINGERS == 'NO':
    YELLOW_FINGERS = 1
else:
    YELLOW_FINGERS = 0

ANXIETY = st.selectbox('Apakah pasien mempunyai kecemasan berlebih?', ['YES', 'NO'])
    
if ANXIETY == 'YES':
    ANXIETY = 2
elif ANXIETY == 'NO':
    ANXIETY = 1
else:
    ANXIETY = 0

PEER_PRESSURE = st.selectbox('Apakah pasien mempunyai tekanan?', ['YES', 'NO'])
    
if PEER_PRESSURE == 'YES':
   PEER_PRESSURE = 2
elif PEER_PRESSURE == 'NO':
    PEER_PRESSURE = 1
else:
    PEER_PRESSURE = 0

COUGHING = st.selectbox('Apakah pasien batuk-batuk?', ['YES', 'NO'])
    
if COUGHING == 'YES':
   COUGHING = 2
elif COUGHING == 'NO':
    COUGHING = 1
else:
    COUGHING = 0

SHORTNESS_OF_BREATH = st.selectbox('Apakah pasien sesak nafas?', ['YES', 'NO'])
    
if SHORTNESS_OF_BREATH== 'YES':
   SHORTNESS_OF_BREATH = 2
elif SHORTNESS_OF_BREATH == 'NO':
    SHORTNESS_OF_BREATH = 1
else:
    SHORTNESS_OF_BREATH = 0

SWALLOWING_DIFFICULTY = st.selectbox('Apakah pasien kesulitan menelan?', ['YES', 'NO'])
    
if SWALLOWING_DIFFICULTY== 'YES':
   SWALLOWING_DIFFICULTY = 2
elif SWALLOWING_DIFFICULTY == 'NO':
    SWALLOWING_DIFFICULTY = 1
else:
    SWALLOWING_DIFFICULTY = 0

CHEST_PAIN = st.selectbox('Apakah pasien nyeri dada?', ['YES', 'NO'])
    
if CHEST_PAIN== 'YES':
   CHEST_PAIN = 2
elif CHEST_PAIN == 'NO':
    CHEST_PAIN = 1
else:
    CHEST_PAIN = 0

CHRONIC_DISEASE = st.selectbox('Apakah pasien mempunyai penyakit kronis?', ['YES', 'NO'])
    
if CHRONIC_DISEASE== 'YES':
   CHRONIC_DISEASE = 2
elif CHRONIC_DISEASE == 'NO':
    CHRONIC_DISEASE = 1
else:
    CHRONIC_DISEASE = 0

WHEEZING = st.selectbox('Apakah pasien mengi (Napas Berbunyi)?', ['YES', 'NO'])
    
if WHEEZING== 'YES':
   WHEEZING = 2
elif WHEEZING == 'NO':
    WHEEZING = 1
else:
    WHEEZING = 0



kelas = ''

if st.button('Estimasi'):
    estimasi_pasien = model.predict([[AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN, CHRONIC_DISEASE, WHEEZING]])
    
    if(estimasi_pasien[0] == 0):
        kanker = 'Pasien menderita kanker paru-paru'
    else :
        kanker ='Pasien tidak menderita kanker paru-paru'

    st.success(kanker)