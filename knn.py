import pickle 
import streamlit as st 

model = pickle.load(open('knn-paru-paru.sav', 'rb'))

st.title('Estimasi Pasien Yang Menderita Kanker Paru-Paru')

col1, col2 = st.columns(2)

with col1 :
    AGE = st.number_input('Input umur pasien')

with col2 :
    SMOKING = st.number_input('Apakah pasien merokok?')

with col1 :
    YELLOW_FINGERS = st.number_input('Apakah pasien jari pasien kuning?')

with col2:
    ANXIETY = st.number_input('Apakah pasien mempunyai kecemasan berlebih?')

with col1 :
    PEER_PRESSURE = st.number_input('Apakah pasien mempunyai tekanan dari teman sebaya?')

with col2 :
    COUGHING = st.number_input('Apakah pasien batuk-batuk?')

with col1 :
    SHORTNESS_OF_BREATH = st.number_input('Apakah pasien sesak nafas?')

with col2 :
    SWALLOWING_DIFFICULTY = st.number_input('Apakah pasien kesulitan menelan?')

with col1 :
    CHEST_PAIN = st.number_input('Apakah pasien nyeri dada?')

with col2 :
    CHRONIC_DISEASE = st.number_input('Apakah pasien mempunyai penyakit kronis?')

with col1 :
    WHEEZING = st.number_input('Apakah pasien mengi (Napas Berbunyi)?')

predict = ''

if st.button('Estimasi ', type="primary"):
    predict = model.predict(
        [[AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN, CHRONIC_DISEASE, WHEEZING]]
    )
    st.write('Apakah orang-orang dengan karakteristik tersebut memiliki kanker paru-paru atau tidak? : ', predict)
