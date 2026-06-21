# 🎯 AI-Powered Face Detection System

An advanced real-time Face Detection System built with Python and OpenCV that detects faces, eyes, and smiles using Haar Cascade Classifiers. The system includes face tracking, automatic face snapshot capturing, screenshot management, FPS monitoring, and a modern UI overlay.

## 📌 Features

### 👤 Real-Time Face Detection

* Detects human faces from a webcam feed
* Draws stylish bounding boxes around detected faces
* Assigns unique IDs to tracked faces

### 👀 Eye Detection

* Detects eyes within detected faces
* Toggle eye detection on/off during runtime

### 😊 Smile Detection

* Detects smiles inside detected face regions
* Toggle smile detection on/off

### 🎯 Face Tracking

* Uses a Centroid Tracking algorithm
* Maintains stable IDs for detected faces
* Tracks faces across multiple frames

### 📸 Screenshot Capture

* Save full-frame screenshots instantly
* Screenshots are automatically stored in the project directory

### 🤖 Auto Face Snapshot Capture

* Automatically captures cropped face images
* Saves snapshots periodically when enabled
* Organizes snapshots by face ID

### 📊 Performance Monitoring

* Real-time FPS calculation
* Detection statistics
* System status monitoring

### 🎨 Modern UI Overlay

* Professional dashboard-style interface
* Notification system
* Color-coded status indicators
* Fullscreen support

---

# 🏗️ Project Structure

```text
AI-Powered-Face-Detection-System/
│
├── main.py
├── detector.py
├── renderer.py
├── capture_manager.py
├── config.py
│
├── haarcascade_frontalface_default.xml
├── haarcascade_eye.xml
├── haarcascade_smile.xml
│
├── captures/
│   ├── screenshots/
│   └── snapshots/
│
├── face_detection.log
│
└── README.md
```

---

# ⚙️ Technologies Used

* Python 3.x
* OpenCV
* NumPy
* Haar Cascade Classifiers
* Computer Vision
* Object Tracking

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Powered-Face-Detection-System.git

cd AI-Powered-Face-Detection-System
```

## 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install opencv-python numpy
```

---

# ▶️ Running the Application

```bash
python main.py
```

After launching:

* Webcam opens automatically
* Face detection begins in real-time
* Tracking IDs appear over detected faces

---

# ⌨️ Keyboard Controls

| Key     | Function                 |
| ------- | ------------------------ |
| Q / ESC | Exit Application         |
| S       | Save Screenshot          |
| E       | Toggle Eye Detection     |
| M       | Toggle Smile Detection   |
| C       | Toggle Auto Face Capture |
| F       | Toggle Fullscreen Mode   |

---

# 📂 Output Files

## Screenshots

Saved to:

```text
captures/screenshots/
```

Example:

```text
screenshot_20260614_120530.jpg
```

## Face Snapshots

Saved to:

```text
captures/snapshots/
```

Example:

```text
face_3_20260614_120535.jpg
```

---

# 📊 Detection Pipeline

```text
Webcam Feed
      │
      ▼
Face Detection
      │
      ▼
Eye Detection
      │
      ▼
Smile Detection
      │
      ▼
Face Tracking
      │
      ▼
UI Rendering
      │
      ▼
Screenshot / Snapshot Saving
```

---

# 🔧 Configuration

All configurable parameters are stored inside:

```python
config.py
```

You can modify:

* Camera resolution
* FPS settings
* Detection sensitivity
* Bounding box colors
* Font sizes
* Auto capture interval
* Tracker settings

---

# 📈 Future Improvements

* Deep Learning Face Detection (DNN)
* Face Recognition
* Attendance Management
* Emotion Detection
* Age & Gender Prediction
* Mask Detection
* Face Analytics Dashboard
* Database Integration

---

# 🛡️ Requirements

```text
Python >= 3.10
OpenCV >= 4.x
NumPy >= 1.24
```

---

# 👨‍💻 Author

**Abdur Rahman**

[![GitHub](https://img.shields.io/badge/GitHub-abdurrahmancce-181717?style=for-the-badge&logo=github)](https://github.com/abdurrahmancce)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Abdur%20Rahman-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/abdur-rahman-akash26/)

---

# ⭐ Support

If you found this project useful:

* Star the repository
* Fork the project
* Share it with others
* Contribute improvements

---

## 📜 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project for educational and personal purposes.

