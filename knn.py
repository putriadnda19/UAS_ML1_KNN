import pickle 
import streamlit as st 

model = pickle.load(open('knn-paru-paru.sav', 'rb'))

st.title('Estimasi Pasien Yang Menderita Kanker Paru-Paru')

AGE = st.number_input('Input umur pasien')
SMOKING = st.number_input('Apakah pasien merokok?')
YELLOW_FINGERS = st.number_input('Apakah pasien jari pasien kuning?')
ANXIETY = st.number_input('Apakah pasien mempunyai kecemasan berlebih?')
PEER_PRESSURE = st.number_input('Apakah pasien mempunyai tekanan dari teman sebaya?')
COUGHING = st.number_input('Apakah pasien batuk-batuk?')
SHORTNESS_OF_BREATH = st.number_input('Apakah pasien sesak nafas?')
SWALLOWING_DIFFICULTY = st.number_input('Apakah pasien kesulitan menelan?')
CHEST_PAIN = st.number_input('Apakah pasien nyeri dada?')
CHRONIC_DISEASE = st.number_input('Apakah pasien mempunyai penyakit kronis?')
WHEEZING = st.number_input('Apakah pasien mengi (Napas Berbunyi)?')

predict = ''

if st.button('Estimasi '):
    predict = model.predict(
        [[AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN, CHRONIC_DISEASE, WHEEZING]]
    )
    st.write('Apakah orang-orang dengan karakteristik tersebut memiliki kanker paru-paru atau tidak? : ', predict)