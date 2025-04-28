import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2


# --- Model Definition (unchanged) ---
class MaskCNN(nn.Module):
    def __init__(self):
        super(MaskCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.flatten_dim = 64 * 30 * 30
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


MODEL_PATH = "best_mask_cnn.pth"


@st.cache_resource
def load_model():
    """Load the trained model once and reuse."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


# — Sidebar —
st.sidebar.title("🔧 Controls")
mode = st.sidebar.radio(
    "Input mode", ["Upload Image", "Take Photo", "Live Camera Stream"]
)
threshold = st.sidebar.slider(
    "Confidence threshold",
    0.0,
    1.0,
    0.5,
    0.01,
    help="Only report a prediction if it exceeds this confidence.",
)
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Mask Detection App**  
    • Powered by a simple CNN.  
    • Upload or snap a photo, or run a live stream.  
    • Adjust the confidence threshold to reduce false alerts.
    """
)

# — Load Model & Setup —
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
model, device = load_model()
labels = ["No Mask", "Mask"]

st.title("😷 Mask Detection")
st.markdown(
    "Check whether someone is wearing a mask.  \n"
    "Choose an input mode from the sidebar, then run inference."
)

input_image = None

# — Static Input Modes —
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])
    if uploaded:
        input_image = Image.open(uploaded).convert("RGB")

elif mode == "Take Photo":
    snap = st.camera_input("Take a photo with your camera")
    if snap:
        input_image = Image.open(snap).convert("RGB")

# — Live Stream Mode —
elif mode == "Live Camera Stream":
    start = st.button("▶️ Start Live Stream")
    stop = st.button("⏹ Stop Stream")
    placeholder = st.empty()

    if start:
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera.")
                break
            # Preprocess & predict
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                probs = F.softmax(out, dim=1).cpu().numpy()[0]
            idx = np.argmax(probs)
            label = labels[idx]
            conf = probs[idx]

            # Annotate
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.putText(
                frame,
                f"{label}: {conf*100:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )
            placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                caption="Live Mask Detection",
            )
        cap.release()
        placeholder.empty()

# — Run Inference on Static Image —
if input_image is not None:
    st.image(input_image, caption="Input Image", width=400)
    with st.spinner("Running inference…"):
        tensor = transform(input_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        idx = np.argmax(probs)
        pred_label = labels[idx]
        pred_conf = probs[idx]

    # Display results
    if pred_conf < threshold:
        st.warning(f"Low confidence ({pred_conf*100:.1f}%), result may be unreliable.")
    else:
        if pred_label == "Mask":
            st.success(f"✅ {pred_label} ({pred_conf*100:.1f}%)")
        else:
            st.error(f"❌ {pred_label} ({pred_conf*100:.1f}%)")

    # Show full probability breakdown
    with st.expander("Show class probabilities"):
        st.write({labels[0]: f"{probs[0]*100:.2f}%", labels[1]: f"{probs[1]*100:.2f}%"})


# Run this in terminal
# streamlit run mask_detection_app.py
