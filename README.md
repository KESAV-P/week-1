# week-1
An AI-based object detection system using the Indian Driving Dataset. It identifies vehicles, pedestrians, and animals from car camera images using YOLOv8, aiming to simulate Tesla-like detection for safer autonomous driving on Indian roads.


# 🚗 Mini Tesla — Object Detection on Indian Roads Using YOLOv8

## 📘 Problem Statement
In real-world road environments, especially in India, roads are filled with various types of vehicles, pedestrians, animals, and obstacles.  
The challenge is to develop an AI model that can **detect and classify multiple objects in real time** from a front-facing car camera, similar to how autonomous vehicles (like Tesla) perceive their surroundings.

## 🎯 Objective
To build and train a **computer vision model** that can detect common road entities such as:
- Cars  
- Trucks  
- Buses  
- Motorcycles  
- Pedestrians  
- Bicycles  
- Animals  

The model should be able to work effectively using dashcam-like input and predict bounding boxes around each detected object.

---

## 💡 Proposed Solution
We use **YOLOv8 (You Only Look Once)** — a modern and efficient object detection algorithm — to train a deep learning model on Indian road scenarios.  
The system is trained using annotated images from **the Indian Driving Dataset (IDD)** that provide realistic road visuals from Hyderabad and Bangalore.

### Steps Followed:
1. **Dataset Preparation**
   - Downloaded and preprocessed the [Indian Driving Dataset (IDD)](https://www.kaggle.com/datasets/manjotpahwa/indian-driving-dataset)
   - Organized into train/validation/test sets
   - Converted annotations into YOLO format (`.txt` files with normalized bounding boxes)

2. **Model Training**
   - Trained YOLOv8 model using the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) library  
   - Tuned parameters like epochs, batch size, and image resolution  
   - Used pretrained weights (`yolov8s.pt`) for faster convergence

3. **Evaluation & Testing**
   - Evaluated model performance using metrics such as precision, recall, and mAP (mean Average Precision)  
   - Tested on unseen Indian road images to verify object detection accuracy

4. **Inference**
   - Performed detection on real dashcam-style images  
   - Visualized results with bounding boxes and class labels

---

## 🧠 Technologies Used
- Python  
- YOLOv8 (Ultralytics)  
- PyTorch  
- OpenCV  
- NumPy, Matplotlib  
- Google Colab / Jupyter Notebook  

---

## 📊 Dataset
**Name:** Indian Driving Dataset (IDD)  
**Source:** [Kaggle - Indian Driving Dataset](https://www.kaggle.com/datasets/manjotpahwa/indian-driving-dataset)  
**Description:**  
- Contains **10,000+ front-facing camera images** captured in Hyderabad and Bangalore  
- Annotated for **34 object classes**, including vehicles, pedestrians, and road objects  
- Ideal for **autonomous driving and object detection tasks**

---

