# Import Streamlit dan pustaka yang diperlukan
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
#import io
from PIL import Image  # Tambahkan ini untuk mengimpor Image dari PIL
from streamlit_autorefresh import st_autorefresh

# Load favicon image
im = Image.open("favicon.ico")
st.set_page_config(
    page_title="UHTP Smart Fire Prediction",
    page_icon=im,

)


# Fungsi untuk mengonversi hari ke bahasa Indonesia
def convert_day_to_indonesian(day_name):
    days_translation = {
        'Monday': 'Senin',
        'Tuesday': 'Selasa',
        'Wednesday': 'Rabu',
        'Thursday': 'Kamis',
        'Friday': 'Jumat',
        'Saturday': 'Sabtu',
        'Sunday': 'Minggu'
    }
    return days_translation.get(day_name, day_name)

# Fungsi untuk mengonversi bulan ke bahasa Indonesia
def convert_month_to_indonesian(month_name):
    months_translation = {
        'January': 'Januari',
        'February': 'Februari',
        'March': 'Maret',
        'April': 'April',
        'May': 'Mei',
        'June': 'Juni',
        'July': 'Juli',
        'August': 'Agustus',
        'September': 'September',
        'October': 'Oktober',
        'November': 'November',
        'December': 'Desember'
    }
    return months_translation.get(month_name, month_name)

# Menambahkan logo di sebelah kiri tulisan "UHTP Smart Fire Prediction"
col1, col2 = st.columns([1, 6])  # Membuat layout kolom untuk logo dan judul
with col1:
    st.image("logo.png", width=100)  # Menambahkan logo dari folder yang sama dengan aplikasi
with col2:
    # Judul Aplikasi
    st.title("UHTP Smart Fire Prediction")

# Deskripsi Aplikasi
st.markdown("""
Sistem Prediksi Tingkat Resiko Kebakaran Hutan dan Lahan menggunakan pengembangan model Hybrid Machine dan Deep Learning.
Data diambil dari perangkat IoT secara Realtime [Google Sheets](https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/edit?usp=sharing).
""")

# Fungsi untuk memuat data
@st.cache_data
def load_data(url):
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        return None

# Fungsi untuk memuat model
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# Fungsi untuk memuat scaler yang sudah dilatih
@st.cache_resource
def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat scaler: {e}")
        return None

# URL Data Google Sheets (format CSV)
data_url = 'https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv'

# Tombol untuk refresh data
#if st.button('Refresh Data'):
#    st.cache_data.clear()  # Hapus cache agar data terbaru dimuat

# Refresh otomatis setiap 3 detik
st_autorefresh(interval=3000, limit=None, key="data_refresh")
st.cache_data.clear()

# Muat Data
sensor_data = load_data(data_url)

# Muat Model dan Scaler
model = load_model('meta_LR.joblib')
scaler = load_scaler('scaler.joblib')

# Tampilkan hasil prediksi data paling akhir sebelum data sensor
if sensor_data is not None and model is not None and scaler is not None:
    st.subheader("Hasil Prediksi Data Realtime")

    # Mengganti nama kolom sesuai dengan model yang dilatih
    sensor_data = sensor_data.rename(columns={
        'Suhu Udara': 'Tavg: Temperatur rata-rata (°C)',
        'Kelembapan Udara': 'RH_avg: Kelembapan rata-rata (%)',
        'Curah Hujan/Jam': 'RR: Curah hujan (mm)',
        'Kecepatan Angin (ms)': 'ff_avg: Kecepatan angin rata-rata (m/s)',
        'Kelembapan Tanah': 'Kelembaban Perbukaan Tanah',
        'Waktu': 'Waktu'  # Pastikan ada kolom waktu
    })

    # Fitur yang akan diprediksi
    fitur = ['Tavg: Temperatur rata-rata (°C)', 'RH_avg: Kelembapan rata-rata (%)', 'RR: Curah hujan (mm)',
             'ff_avg: Kecepatan angin rata-rata (m/s)', 'Kelembaban Perbukaan Tanah']

    if all(col in sensor_data.columns for col in fitur):
        fitur_data = sensor_data[fitur]

        # Mengganti koma dengan titik agar bisa dikonversi ke float
        for col in fitur_data.columns:
            fitur_data[col] = fitur_data[col].astype(str).str.replace(',', '.').astype(float)

        # Mengisi nilai yang hilang dengan 0
        fitur_data.fillna(0, inplace=True)

        # Standarisasi Fitur menggunakan scaler yang sudah dilatih
        fitur_scaled = scaler.transform(fitur_data)

        # Buat DataFrame dari fitur yang sudah di-scale
        fitur_scaled_df = pd.DataFrame(fitur_scaled, columns=fitur)

        # Prediksi
        predictions = model.predict(fitur_scaled_df)

        # Konversi prediksi numerik ke label kategori
        def convert_to_label(pred):
            if pred == 0:
                return "High / Tinggi"
            elif pred == 1:
                return "Low / Rendah"
            elif pred == 2:
                return "Moderate / Sedang"
            elif pred == 3:
                return "Very High / Sangat Tinggi"
            else:
                return "Unknown"

        sensor_data['Prediksi Kebakaran'] = [convert_to_label(pred) for pred in predictions]

        # Mengambil waktu dari kolom waktu dan format menjadi hari, tanggal, bulan, tahun
        last_row = sensor_data.iloc[-1]
        waktu_prediksi = pd.to_datetime(last_row['Waktu'])
        hari_indonesia = convert_day_to_indonesian(waktu_prediksi.strftime('%A'))
        bulan_indonesia = convert_month_to_indonesian(waktu_prediksi.strftime('%B'))
        tanggal_prediksi = waktu_prediksi.strftime(f'%d {bulan_indonesia} %Y')

        # Menampilkan data sensor tanpa kolom indeks, dengan header rata tengah dan warna abu-abu
        st.write("**Data Sensor Realtime:**")
        st.markdown("""
            <style>
            table { width: 100%; }
            thead th { text-align: center; background-color: #f0f0f0; }
            td { text-align: left; }
            th { width: 40%; }  /* Mengatur lebar kolom Variabel */
            </style>
        """, unsafe_allow_html=True)

        sensor_html = pd.DataFrame({
            "Variabel": ["Tavg: Temperatur rata-rata (°C)", "RH_avg: Kelembapan rata-rata (%)", "RR: Curah hujan (mm)", "ff_avg: Kecepatan angin rata-rata (m/s)", "Kelembaban Perbukaan Tanah"],
            "Value": last_row[fitur].values
        }).to_html(index=False)

        st.markdown(sensor_html, unsafe_allow_html=True)

        # Prediksi Kebakaran berdasarkan risiko
        risk = last_row['Prediksi Kebakaran']
        risk_styles = {
            "Low / Rendah": {"color": "white", "background-color": "blue"},
            "Moderate / Sedang": {"color": "white", "background-color": "green"},
            "High / Tinggi": {"color": "black", "background-color": "yellow"},
            "Very High / Sangat Tinggi": {"color": "white", "background-color": "red"}
        }

        risk_style = risk_styles.get(risk, {"color": "black", "background-color": "white"})

        # Menampilkan prediksi kebakaran dengan indikator risiko lebih besar, tebal, dan garis bawah
        st.markdown(
            f"<p style='color:{risk_style['color']}; background-color:{risk_style['background-color']}; padding: 10px; border-radius: 5px;'>"
            f"Pada hari {hari_indonesia}, tanggal {tanggal_prediksi}, lahan ini diprediksi memiliki tingkat resiko kebakaran: "
            f"<span style='font-weight: bold; font-size: 28px; text-decoration: underline;'>{risk}</span></p>", 
            unsafe_allow_html=True
        )

    else:
        st.error("Data sensor tidak memiliki semua kolom fitur yang diperlukan.")


# Penjelasan tingkat resiko kebakaran
st.markdown("""
    **Tabel berikut menunjukkan besarnya tingkat resiko kebakaran dan intensitas api jika terjadi kebakaran hutan dan lahan.**
""")

# Fire risk intensity table
st.markdown("""
    <table style="width:100%; border-collapse: collapse;">
        <thead>
            <tr style="background-color: #f0f0f0; text-align: left;">
                <th style="padding: 8px; border: 1px solid #ddd; width: 15%;">Warna</th>
                <th style="padding: 8px; border: 1px solid #ddd; width: 20%;">Tingkat Resiko / Intensitas</th>
                <th style="padding: 8px; border: 1px solid #ddd; width: 65%;">Keterangan</th>
            </tr>
        </thead>
        <tbody>
            <tr style="background-color: blue; color: white;">
                <td style="padding: 8px; border: 1px solid #ddd;">Blue</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Low</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Tingkat resiko kebakaran rendah. Intensitas api pada kategori rendah. Api mudah dikendalikan, cenderung akan padam dengan sendirinya.</td>
            </tr>
            <tr style="background-color: green; color: white;">
                <td style="padding: 8px; border: 1px solid #ddd;">Green</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Moderate</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Tingkat resiko kebakaran sedang. Intensitas api pada kategori sedang. Api relatif masih cukup mudah dikendalikan.</td>
            </tr>
            <tr style="background-color: yellow; color: black;">
                <td style="padding: 8px; border: 1px solid #ddd;">Yellow</td>
                <td style="padding: 8px; border: 1px solid #ddd;">High</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Tingkat resiko kebakaran tinggi. Intensitas api pada kategori tinggi. Api sulit dikendalikan.</td>
            </tr>
            <tr style="background-color: red; color: white;">
                <td style="padding: 8px; border: 1px solid #ddd;">Red</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Very High</td>
                <td style="padding: 8px; border: 1px solid #ddd;">Tingkat resiko kebakaran sangat tinggi. Intensitas api pada kategori sangat tinggi. Api sangat sulit dikendalikan.</td>
            </tr>
        </tbody>
    </table>
""", unsafe_allow_html=True)





# Bagian Data Sensor di bawah Hasil Prediksi
if sensor_data is not None:
    st.subheader("Data Sensor")
    st.dataframe(sensor_data)

    # Fitur download hasil prediksi sebagai CSV
    csv = sensor_data.to_csv(index=False)
    st.download_button(
        label="Download Hasil Prediksi sebagai CSV",
        data=csv,
        file_name='hasil_prediksi_kebakaran.csv',
        mime='text/csv'
    )

# Fitur Input Manual untuk Prediksi Real-time
if model is not None and scaler is not None:
    st.subheader("Prediksi Kebakaran Baru")
    st.markdown("Masukkan nilai sensor untuk memprediksi kemungkinan kebakaran.")

    suhu = st.number_input("Suhu Udara (°C)", min_value=0.0, max_value=100.0, value=25.0)
    kelembapan_udara = st.number_input("Kelembapan Udara (%)", min_value=0.0, max_value=100.0, value=50.0)
    curah_hujan = st.number_input("Curah Hujan/Jam (mm)", min_value=0.0, max_value=500.0, value=10.0)
    kecepatan_angin = st.number_input("Kecepatan Angin (ms)", min_value=0.0, max_value=100.0, value=5.0)
    kelembapan_tanah = st.number_input("Kelembapan Tanah (%)", min_value=0.0, max_value=100.0, value=40.0)

    # Buat DataFrame dari input pengguna
    input_data = pd.DataFrame({
        'Tavg: Temperatur rata-rata (°C)': [suhu],
        'RH_avg: Kelembapan rata-rata (%)': [kelembapan_udara],
        'RR: Curah hujan (mm)': [curah_hujan],
        'ff_avg: Kecepatan angin rata-rata (m/s)': [kecepatan_angin],
        'Kelembaban Perbukaan Tanah': [kelembapan_tanah]
    })

    # Pra-pemrosesan input pengguna menggunakan scaler yang sudah dilatih
    input_scaled = scaler.transform(input_data)

    # Prediksi untuk input pengguna
    user_prediction = model.predict(input_scaled)
    user_label = convert_to_label(user_prediction[0])

    # Menampilkan hasil prediksi dengan background warna
    user_risk_style = risk_styles.get(user_label, {"color": "black", "background-color": "white"})

    st.markdown(
        f"<p style='color:{user_risk_style['color']}; background-color:{user_risk_style['background-color']}; font-weight: bold; padding: 10px; border-radius: 5px;'>Prediksi Risiko Kebakaran: {user_label}</p>", 
        unsafe_allow_html=True
    )

# Footer dengan logo dan tulisan
st.markdown("---")  # Garis pembatas untuk memisahkan footer
col1, col2, col3 = st.columns([1, 3, 1])  # Layout kolom untuk gambar logo dan teks
with col1:
    st.image("kemdikbud.png", width=100)  # Menampilkan logo Kemdikbud
with col2:
    st.markdown("<h3 style='text-align: center;'>UHTP Smart Fire Prediction V1</h3>", unsafe_allow_html=True)
with col3:
    st.image("uhtp.png", width=100)  # Menampilkan logo UHTP

st.markdown("<p style='text-align: center;'>Dikembangkan oleh Tim Dosen Universitas Hang Tuah Pekanbaru Tahun 2024</p>", unsafe_allow_html=True)
