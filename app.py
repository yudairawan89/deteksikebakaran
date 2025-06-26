import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from timm import create_model
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
from ultralytics import YOLO
import pandas as pd
import joblib
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ============ Konfigurasi Tampilan Streamlit ============
st.set_page_config(page_title="🔥 Deteksi Api dan Klasifikasi", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: red;'>🔥 Sistem Deteksi Kebakaran</h1>
    <h4 style='text-align: center;'>Deteksi Lokasi Api dengan YOLO + Klasifikasi Gambar Menggunakan ViT-GRU</h4>
    <hr style="border: 2px solid red;">
""", unsafe_allow_html=True)

# ============ Load Model ============
yolo_model = YOLO("best.pt")
vit_gru_model = None

# === Fungsi konversi bahasa Indonesia ===
def convert_day_to_indonesian(day_name):
    return {'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
            'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu',
            'Sunday': 'Minggu'}.get(day_name, day_name)

def convert_month_to_indonesian(month_name):
    return {'January': 'Januari', 'February': 'Februari', 'March': 'Maret',
            'April': 'April', 'May': 'Mei', 'June': 'Juni', 'July': 'Juli',
            'August': 'Agustus', 'September': 'September', 'October': 'Oktober',
            'November': 'November', 'December': 'Desember'}.get(month_name, month_name)

def convert_to_label(pred):
    return {0: "Low / Rendah", 1: "Moderate / Sedang", 2: "High / Tinggi", 3: "Very High / Sangat Tinggi"}.get(pred, "Unknown")

risk_styles = {
    "Low / Rendah": ("white", "blue"),
    "Moderate / Sedang": ("white", "green"),
    "High / Tinggi": ("black", "yellow"),
    "Very High / Sangat Tinggi": ("white", "red")
}

# === Load Model & Scaler ===
@st.cache_resource
def load_model():
    return joblib.load("RHSEM_IoT_Model.joblib")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.joblib")

model = load_model()
scaler = load_scaler()

# === Load Data Sensor dari Google Sheets ===
@st.cache_data(ttl=60)
def load_sensor_data():
    url = "https://docs.google.com/spreadsheets/d/1ZscUJ6SLPIF33t8ikVHUmR68b-y3Q9_r_p9d2rDRMCM/export?format=csv"
    return pd.read_csv(url)

# === PREDIKSI DARI SENSOR REALTIME ===
with st.container():
    st_autorefresh(interval=7000, key="refresh_iot")
    df = load_sensor_data()

    if df is not None and not df.empty:
        df = df.rename(columns={
            'Suhu Udara': 'Tavg: Temperatur rata-rata (°C)',
            'Kelembapan Udara': 'RH_avg: Kelembapan rata-rata (%)',
            'Curah Hujan/Jam': 'RR: Curah hujan (mm)',
            'Kecepatan Angin (ms)': 'ff_avg: Kecepatan angin rata-rata (m/s)',
            'Kelembapan Tanah': 'Kelembaban Permukaan Tanah',
            'Waktu': 'Waktu'
        })

        fitur = [
            'Tavg: Temperatur rata-rata (°C)',
            'RH_avg: Kelembapan rata-rata (%)',
            'RR: Curah hujan (mm)',
            'ff_avg: Kecepatan angin rata-rata (m/s)',
            'Kelembaban Permukaan Tanah'
        ]

        clean_df = df[fitur].copy()
        for col in fitur:
            clean_df[col] = clean_df[col].astype(str).str.replace(',', '.').astype(float).fillna(0)

        scaled_all = scaler.transform(clean_df)
        predictions = [convert_to_label(p) for p in model.predict(scaled_all)]
        df["Prediksi Kebakaran"] = predictions

        last_row = df.iloc[-1]
        waktu = pd.to_datetime(last_row['Waktu'])
        hari = convert_day_to_indonesian(waktu.strftime('%A'))
        bulan = convert_month_to_indonesian(waktu.strftime('%B'))
        tanggal = waktu.strftime(f'%d {bulan} %Y')
        risk_label = last_row["Prediksi Kebakaran"]
        font, bg = risk_styles.get(risk_label, ("black", "white"))

        sensor_df = pd.DataFrame({
            "Variabel": fitur,
            "Value": [f"{last_row[col]:.1f}" for col in fitur]
        })

        st.markdown("<h5 style='text-align: center;'>Data Sensor Realtime</h5>", unsafe_allow_html=True)
        sensor_html = "<table style='width: 100%; border-collapse: collapse;'>"
        sensor_html += "<thead><tr><th>Variabel</th><th>Value</th></tr></thead><tbody>"
        for i in range(len(sensor_df)):
            var = sensor_df.iloc[i, 0]
            val = sensor_df.iloc[i, 1]
            sensor_html += f"<tr><td style='padding:6px;'>{var}</td><td style='padding:6px;'>{val}</td></tr>"
        sensor_html += "</tbody></table>"
        st.markdown(sensor_html, unsafe_allow_html=True)

        st.markdown(
            f"<p style='background-color:{bg}; color:{font}; padding:10px; border-radius:8px; font-weight:bold;'>"
            f"Pada hari {hari}, tanggal {tanggal}, lahan ini diprediksi memiliki tingkat resiko kebakaran: "
            f"<span style='text-decoration: underline; font-size: 22px;'>{risk_label}</span></p>",
            unsafe_allow_html=True
        )





import gdown

# ======== Cek dan Unduh vitgru.pt dari Google Drive jika belum ada ========
vitgru_path = "vitgru.pt"
gdrive_url = "https://drive.google.com/uc?id=18L1CzKDuz-ESnJUdzlOkKYl2GbPA2gvI"

if not os.path.exists(vitgru_path):
    with st.spinner("Mengunduh model ViT+GRU dari Google Drive..."):
        gdown.download(gdrive_url, vitgru_path, quiet=False)


@st.cache_resource
def load_vit_gru():
    class ViT_GRU(torch.nn.Module):
        def _init_(self, hidden_dim=128, num_classes=2):
            super(ViT_GRU, self)._init_()
            self.vit = create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
            self.vit.head = torch.nn.Identity()
            self.gru = torch.nn.GRU(768, hidden_dim, batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            with torch.no_grad():
                vit_feat = self.vit(x).unsqueeze(1)
            out, _ = self.gru(vit_feat)
            return self.fc(out[:, -1, :])

    model = ViT_GRU()
    model.load_state_dict(torch.load("vitgru.pt", map_location="cpu"))
    model.eval()
    return model

vit_gru_model = load_vit_gru()

# ============ Fungsi Transformasi Gambar ============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============ Fungsi Klasifikasi ViT-GRU ============

def classify_fire(image):
    tensor = transform(image).unsqueeze(0)
    outputs = vit_gru_model(tensor)
    prob = torch.softmax(outputs, dim=1)
    label = torch.argmax(prob).item()

    # Membalik label (0 jadi 1, 1 jadi 0)
    label = 1 - label
    return label, prob[0][1 - label].item()


# ============ Fungsi Deteksi YOLO ============
def detect_fire_yolo(img_pil):
    img_array = np.array(img_pil)
    results = yolo_model(img_array, verbose=False)[0]
    boxes = results.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{results.names[cls]} ({conf*100:.1f}%)"
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_array, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return Image.fromarray(img_array)

# ============ Upload atau Kamera ============
st.sidebar.header("📸 Input Gambar")
option = st.sidebar.radio("Pilih metode input", ["Upload Gambar", "Gunakan Kamera"])

if option == "Upload Gambar":
    uploaded = st.sidebar.file_uploader("Upload file gambar", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
elif option == "Gunakan Kamera":
    image = st.camera_input("Ambil gambar dengan kamera")
    if image:
        image = Image.open(image).convert("RGB")

# ============ Tampilkan Hasil Deteksi ============
if 'image' in locals():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟥 Hasil Deteksi Bounding Box (YOLO)")
        detected_image = detect_fire_yolo(image.copy())
        st.image(detected_image, use_column_width=True)

    with col2:
        st.subheader("🔥 Hasil Klasifikasi Gambar (ViT-GRU)")
        label, confidence = classify_fire(image)
        label_str = "FIRE 🔥" if label == 1 else "NON-FIRE ✅"
        color = "red" if label == 1 else "green"
        st.markdown(f"<h2 style='text-align: center; color: {color};'>{label_str}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence*100:.2f}%</b></p>", unsafe_allow_html=True)
else:
    st.info("Silakan upload atau ambil gambar terlebih dahulu.")
