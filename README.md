# Smart Waste Classification System

## Project Overview
This project implements a **real-time waste classification system** using a **ResNet101V2** deep learning model.  
It classifies different types of waste and provides **voice-based disposal instructions** to promote **efficient and eco-friendly waste management**.

---

## Features

- **Classify waste types** in real time using a pre-trained **ResNet101V2** deep learning model.  
- **Real-time video classification** using **OpenCV** for frame-by-frame prediction.
- **Voice assistant** announces the detected waste type and instruct the proper disposal bin.
- **Enhance environmental awareness** and encourage **sustainable waste management**.  
- **Deliver fast and accurate predictions** suitable for real-time deployment.  

---

## Dataset
Garbage Dataset: **https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2**

### Bin Instruction Mapping
| Waste Type | Recommended Bin                              |
| ---------- | -------------------------------------------- |
| paper      | Paper recycling bin                          |
| plastic    | Plastic recycling bin                        |
| shoes      | Reusable items bin or Textile donation box    |
| cardboard  | Paper recycling bin                          |
| clothes    | Textile recycling bin or Donation box         |
| metal      | Metal recycling bin                          |
| trash      | General waste bin                            |
| biological | Organic compost bin                          |
| glass      | Glass recycling bin                          |
| battery    | Hazardous waste bin |

  ---

## Model Overview

- **Base Model:** ResNet101V2 (Transfer Learning)
- **Framework:** TensorFlow / Keras
- **Input Shape:** (224 × 224 × 3)

---

## Model Training Configuration

| Parameter | Value |
|------------|--------|
| Optimizer | AdamW |
| Loss Function | Categorical Crossentropy |
| Epochs | 20 |
| Batch Size | 32 |
| Evaluation Metric | Accuracy |

---

## Dependencies

- **tensorflow**
- **opencv-python**
- **numpy** 
- **pyttsx3**
- **matplotlib**
