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

# ============ Konfigurasi Tampilan Streamlit ============
st.set_page_config(page_title="🔥 Deteksi Api dan Klasifikasi", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: red;'>🔥 Sistem Deteksi Kebakaran</h1>
    <h4 style='text-align: center;'>Deteksi Lokasi Api dengan YOLO + Klasifikasi Gambar Menggunakan ViT-GRU</h4>
    <hr style="border: 2px solid red;">
""", unsafe_allow_html=True)

# ============ Load Model ============
yolo_model = YOLO("best.torchscript")
vit_gru_model = None



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
        def __init__(self, hidden_dim=128, num_classes=2):
            super(ViT_GRU, self).__init__()
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
