import pickle 
import streamlit as st 

model = pickle.load(open('knn-paru-paru.sav', 'rb'))

st.title('Estimasi Pasien Yang Menderita Kanker Paru-Paru')

AGE = st.number_input('**Input umur pasien**')

SMOKING = st.radio('**Apakah pasien merokok?**', ["1", "2")
st.write('**Note**: 1 = NO, 2 = YES')

YELLOW_FINGERS = st.radio('**Apakah pasien jari pasien kuning?**', ["1", "2"])
st.write('**Note**: 1 = NO, 2 = YES')

ANXIETY = st.radio('**Apakah pasien mempunyai kecemasan berlebih?**', ["1", "2"])
st.write('**Note**: 1 = NO, 2 = YES')

PEER_PRESSURE = st.radio('**Apakah pasien mempunyai tekanan dari teman sebaya?**', ["1", "2"])
st.write('**Note**: 1 = NO, 2 = YES')

COUGHING = st.radio('**Apakah pasien batuk-batuk?**', ["1", "2"])
st.write('**Note**: 1 = NO, 2 = YES')

SHORTNESS_OF_BREATH = st.radio('**Apakah pasien sesak nafas?**', ["1", "2"])
st.write('**Note**: 1 = NO, 2 = YES')

SWALLOWING_DIFFICULTY = st.radio('**Apakah pasien kesulitan menelan?**', ["1", "2"])
st.write('**Note**: 1 = NO, 2 = YES')

CHEST_PAIN = st.radiot('**Apakah pasien nyeri dada?**', ["1", "2"])
st.write('**Note**: 1 = NO, 2 = YES')

CHRONIC_DISEASE = st.radio('**Apakah pasien mempunyai penyakit kronis?**', ["1", "2"])
st.write('**Note**: 1 = NO, 2 = YES')

WHEEZING = st.radio('**Apakah pasien mengi (Napas Berbunyi)?**', ["1", "2"])
st.write('**Note**: 1 = NO, 2 = YES')

predict = ''

if st.button('Estimasi '):
    predict = model.predict(
        [[AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN, CHRONIC_DISEASE, WHEEZING]]
    )
    st.write('Apakah orang-orang dengan karakteristik tersebut memiliki kanker paru-paru atau tidak? : ', predict)
