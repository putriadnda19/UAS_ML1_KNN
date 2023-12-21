# app.py

import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Fungsi untuk melatih model KNN
def train_knn_model(X_train, y_train, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

# Fungsi untuk mengevaluasi model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main():
    st.title("Estimasi Pasien Yang Menderita Kanker Paru-Paru")

    # Tambahkan elemen-elemen UI
    k_value = st.slider("Umur pasien", 1, 80)
    k_value = st.radio('**Apakah pasien merokok?**', ["1", "2"])
    st.write('**Note**: 1 = NO, 2 = YES')
    
    k_value = st.radio('**Apakah pasien jari pasien kuning?**', ["1", "2"])
    st.write('**Note**: 1 = NO, 2 = YES')
    
    k_value = st.radio('**Apakah pasien mempunyai kecemasan berlebih?**', ["1", "2"])
    st.write('**Note**: 1 = NO, 2 = YES')
    
    k_value = st.radio('**Apakah pasien mempunyai tekanan dari teman sebaya?**', ["1", "2"])
    st.write('**Note**: 1 = NO, 2 = YES')
    
    k_value = st.radio('**Apakah pasien batuk-batuk?**', ["1", "2"])
    st.write('**Note**: 1 = NO, 2 = YES')
    
    k_value = st.radio('**Apakah pasien sesak nafas?**', ["1", "2"])
    st.write('**Note**: 1 = NO, 2 = YES')
    
    k_value = st.radio('**Apakah pasien kesulitan menelan?**', ["1", "2"])
    st.write('**Note**: 1 = NO, 2 = YES')
    
    k_value = st.radio('**Apakah pasien nyeri dada?**', ["1", "2"])
    st.write('**Note**: 1 = NO, 2 = YES')
    
    k_value = st.radio('**Apakah pasien mempunyai penyakit kronis?**', ["1", "2"])
    st.write('**Note**: 1 = NO, 2 = YES')
    
    k_value = st.radio('**Apakah pasien mengi (Napas Berbunyi)?**', ["1", "2"])
    st.write('**Note**: 1 = NO, 2 = YES')

    # Muat dataset (ganti dengan dataset Anda)
    # Misalnya, Anda dapat menggunakan dataset iris untuk contoh
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Pisahkan data menjadi data pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model
    model = train_knn_model(X_train, y_train, n_neighbors=k_value)

    # Evaluasi model
    accuracy = evaluate_model(model, X_test, y_test)

    # Tampilkan hasil evaluasi
    st.subheader("Hasil Evaluasi Model")
    st.write(f"Akurasi: {accuracy:.2%}")

if __name__ == "__main__":
    main()
