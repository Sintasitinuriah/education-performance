import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan scaler
model = joblib.load('models/svm_model_1.joblib')
scaler = joblib.load('models/scaler_1.joblib')

# Daftar fitur yang digunakan oleh model
selected_features = [
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Tuition_fees_up_to_date', 'Scholarship_holder',
    'Application_mode', 'Gender', 'Debtor', 'Age_at_enrollment'
]

st.set_page_config(page_title="Dashboard Prediksi Dropout Mahasiswa", layout="wide")
st.title("🎓 Dashboard Prediksi Dropout Mahasiswa")

# Sidebar untuk input pengguna
st.sidebar.title("🧾 Input Mahasiswa")
user_input = {}

user_input['Curricular_units_1st_sem_approved'] = st.sidebar.number_input("Mata kuliah lulus semester 1", min_value=0, step=1)
user_input['Curricular_units_1st_sem_grade'] = st.sidebar.number_input("Rata-rata nilai semester 1", min_value=0.0, max_value=20.0, step=0.1)
user_input['Curricular_units_2nd_sem_approved'] = st.sidebar.number_input("Mata kuliah lulus semester 2", min_value=0, step=1)
user_input['Curricular_units_2nd_sem_grade'] = st.sidebar.number_input("Rata-rata nilai semester 2", min_value=0.0, max_value=20.0, step=0.1)

user_input['Tuition_fees_up_to_date'] = st.sidebar.selectbox("Pembayaran UKT Lunas?", ("Ya", "Tidak"))
user_input['Scholarship_holder'] = st.sidebar.selectbox("Penerima Beasiswa?", ("Ya", "Tidak"))
user_input['Application_mode'] = st.sidebar.selectbox("Jalur Masuk", [1, 2, 3, 4, 5])  # Sesuaikan dengan data asli
user_input['Gender'] = st.sidebar.selectbox("Jenis Kelamin", ("Laki-laki", "Perempuan"))
user_input['Debtor'] = st.sidebar.selectbox("Memiliki Utang Pendidikan?", ("Ya", "Tidak"))
user_input['Age_at_enrollment'] = st.sidebar.number_input("Usia saat masuk kuliah", min_value=15, max_value=60)

# Mapping input kategori ke numerik
mapped_input = {
    'Curricular_units_1st_sem_approved': user_input['Curricular_units_1st_sem_approved'],
    'Curricular_units_1st_sem_grade': user_input['Curricular_units_1st_sem_grade'],
    'Curricular_units_2nd_sem_approved': user_input['Curricular_units_2nd_sem_approved'],
    'Curricular_units_2nd_sem_grade': user_input['Curricular_units_2nd_sem_grade'],
    'Tuition_fees_up_to_date': 1 if user_input['Tuition_fees_up_to_date'] == "Ya" else 0,
    'Scholarship_holder': 1 if user_input['Scholarship_holder'] == "Ya" else 0,
    'Application_mode': user_input['Application_mode'],
    'Gender': 1 if user_input['Gender'] == "Laki-laki" else 0,
    'Debtor': 1 if user_input['Debtor'] == "Ya" else 0,
    'Age_at_enrollment': user_input['Age_at_enrollment']
}

# Tombol prediksi
if st.sidebar.button("🔍 Prediksi Dropout"):
    input_df = pd.DataFrame([mapped_input])[selected_features]
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    status = "🎓 **Graduate**" if pred == 1 else "⚠️ **Dropout**"
    st.subheader("Hasil Prediksi")
    st.success(f"Status Prediksi Mahasiswa: {status}")

# Bagian Visualisasi Dataset jika tersedia
data_file = st.file_uploader("📂 Upload file data mahasiswa (CSV)", type="csv")

if data_file:
    df = pd.read_csv(data_file)
    print(df['Curricular_units_1st_sem_approved'].head())
    st.subheader("📊 Statistik Dataset")
    st.dataframe(df.head())

    # Plot dropout berdasarkan jumlah mata kuliah lulus semester 1
    st.markdown("### Dropout Berdasarkan Jumlah Mata Kuliah Lulus Semester 1")
    if 'Status' in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='Curricular_units_1st_sem_approved', hue='Status', multiple='stack', ax=ax)
        ax.set_xlabel("Mata Kuliah Lulus Semester 1")
        ax.set_ylabel("Jumlah Mahasiswa")
        st.pyplot(fig)

    # Plot rasio dropout berdasarkan status beasiswa
    st.markdown("### Proporsi Dropout Berdasarkan Status Beasiswa")
    if 'Scholarship_holder' in df.columns and 'Status' in df.columns:
        prop = df.groupby('Scholarship_holder')['Status'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(x='Scholarship_holder', y='Status', data=prop, ax=ax)
        ax.set_ylabel("Proporsi Dropout")
        ax.set_xlabel("Penerima Beasiswa")
        st.pyplot(fig)

    # Tambahkan visualisasi lain sesuai kebutuhan
