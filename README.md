
# Mask Detection App

👉 Try it live on Streamlit Cloud: [https://face-mask-detection-vignesh.streamlit.app](https://face-mask-detection-vignesh.streamlit.app)

A simple Streamlit application that uses a custom CNN to detect whether a person in an image or live video stream is wearing a mask.

---

## 📋 Table of Contents

1. [Features](#features)  
2. [Project Structure](#project-structure)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Model Details](#model-details)  
6. [Configuration & Controls](#configuration--controls)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## 🔍 Features

- **Upload Image**: Test static images (JPG/PNG) from your local machine.  
- **Take Photo**: Snap a photo directly from your webcam.  
- **Live Camera Stream**: Run real-time inference on a live video feed.  
- **Adjustable Confidence Threshold**: Filter out low-confidence predictions.  
- **Probability Breakdown**: View the model’s class-wise probabilities.  
- **GPU/CPU Support**: Automatic device selection via PyTorch.

---

## 🗂 Project Structure

```

.
├── best_mask_cnn.pth         # Pre-trained model weights
├── mask_detection_app.py     # Main Streamlit application
├── requirements.txt          # Python dependencies
└── README.md                 # This documentation

```

---

## ⚙️ Installation

1. **Clone the repository**  

   ```bash
   git clone https://github.com/your-username/face_mask_detection_app.git
   cd face_mask_detection_app
   ```

2. **Create & activate a virtual environment**  

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**  

   ```bash
   pip install -r requirements.txt
   ```

4. **Download / place the model weights**  
   Ensure `best_mask_cnn.pth` is in the project root.

---

## 🚀 Usage

Run the Streamlit app from your terminal:

```bash
streamlit run mask_detection_app.py
```

- Open the URL printed by Streamlit (usually `http://localhost:8501`) in your browser.  
- Use the sidebar to select your input mode & adjust the confidence slider.  
- Follow on-screen prompts to upload, snap, or stream video.

---

## 🧠 Model Details

- **Architecture**:  
  - 2 convolutional layers (32 → 64 filters, kernel 3×3)  
  - MaxPool (2×2) after each conv layer  
  - 2 fully-connected layers (128 → 64) with Dropout (0.5)  
  - Final output layer (2 classes: Mask / No Mask)

- **Input Size**:  
  - Images are resized to 128×128 before tensor conversion.

- **Training**:  
  Trained on a balanced face-mask dataset; weights saved in `best_mask_cnn.pth`.

- **Device**:  
  - Auto-detects GPU (CUDA) or CPU.

---

## 🛠 Configuration & Controls

- **Input Mode**  
  - **Upload Image**: Browse JPG/PNG files.  
  - **Take Photo**: Use your webcam to capture a snapshot.  
  - **Live Camera Stream**: Click ▶️ to start, ⏹ to stop real-time inference.

- **Confidence Threshold**  
  Move the slider (0.0 – 1.0) to ignore predictions below the chosen confidence.

---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m "Add new feature"`)  
4. Push to the branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
