

---

# 🧍‍♂️🧍‍♀️ Shopping Mall / Lift People Counter using YOLOv8

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=yolo&logoColor=white" alt="YOLOv8" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Deep%20Learning-AI-red?logo=tensorflow&logoColor=white" alt="Deep Learning" />
</p>

---

![Demo](demo.gif)  
*A live demo showcasing real-time people detection and counting using YOLOv8.*

---

## 🚀 Project Overview

This project is built to **detect and count people** in crowded environments such as **shopping malls, lifts, or event halls** using the **YOLOv8 (You Only Look Once)** deep learning model.

It processes a video input, identifies each person, and displays **bounding boxes** and a **live counter** overlay. This can be used for **crowd analysis, safety monitoring**, or **building automation systems**.

---

## ✨ Key Features

* 🎯 **High-Accuracy People Detection** with YOLOv8
* 📹 **Real-Time Video Processing** using OpenCV
* 💾 **Auto-Saves Processed Video** to your local folder
* 🔢 **Dynamic People Counter** displayed on-screen
* ⚡ **Optimized for Speed** (CPU/GPU compatible)
* 🧠 **Customizable Model** — adapt for other objects easily

---

## 🧰 Technologies Used

* **Python 3.10+**
* **Ultralytics YOLOv8**
* **OpenCV (cv2)**
* **os**, **shutil** for file handling

---

## ⚙️ How It Works

1. Loads the **YOLOv8 pre-trained model** (`yolov8n.pt`).
2. Reads the **input video** file frame by frame.
3. Detects **persons (class ID 0)** in each frame.
4. Draws **bounding boxes** and updates **real-time people count**.
5. Saves the **output video** with all overlays in the project folder.

---

## 🖥️ Folder Structure

```
📂 People_Counter_Project
│
├── people_counter.py           # Main Python script
├── yolov8n.pt                  # YOLOv8 model weights
├── input_video.mp4             # Original test video
├── output_video.mp4            # Processed result
└── README.md                   # Documentation
```

---

## ▶️ How to Run

1. **Install dependencies**

   ```bash
   pip install ultralytics opencv-python
   ```

2. **Place your files** (`people_counter.py`, `yolov8n.pt`, and `input_video.mp4`) in the same directory.

3. **Run the script**

   ```bash
   python people_counter.py
   ```

4. After processing, a new video `output_video.mp4` will be saved automatically.

---

## 🚀 Future Enhancements

* 🔍 Add **entry/exit direction tracking**
* 📊 Build a **real-time analytics dashboard**
* 🕹️ Enable **overcrowding alerts**
* ☁️ Integrate with **IoT and cloud platforms**
* 🧩 Extend detection to **multi-class tracking** (e.g., luggage, staff, etc.)



---

## 👨‍💻 About the Developer

**Usama Munawar** – Data Scientist | MPhil Scholar | Machine Learning Enthusiast  
Passionate about transforming raw data into meaningful insights and intelligent systems.  

🌍 Connect with me:

[![GitHub](https://img.icons8.com/fluent/48/000000/github.png)](https://github.com/UsamaMunawarr)[![LinkedIn](https://img.icons8.com/color/48/000000/linkedin.png)](https://www.linkedin.com/in/abu--usama)[![YouTube](https://img.icons8.com/?size=50\&id=19318\&format=png)](https://www.youtube.com/@CodeBaseStats)[![Twitter](https://img.icons8.com/color/48/000000/twitter.png)](https://twitter.com/Usama__Munawar?t=Wk-zJ88ybkEhYJpWMbMheg&s=09)[![Facebook](https://img.icons8.com/color/48/000000/facebook-new.png)](https://www.facebook.com/profile.php?id=100005320726463&mibextid=9R9pXO)

---
