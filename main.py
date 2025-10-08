import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

model = tf.keras.models.load_model('waste_classifier_model.h5')
class_names = ['paper', 'plastic', 'shoes', 'cardboard', 'clothes', 'metal', 'trash', 'biological', 'glass', 'battery']

bin_dict = {
    'paper': 'Paper recycling bin',
    'plastic': 'Plastic recycling bin',
    'shoes': 'Reusable items bin or textile donation box',
    'cardboard': 'Paper recycling bin',
    'clothes': 'Textile recycling bin or donation box',
    'metal': 'Metal recycling bin',
    'trash': 'General waste bin',
    'biological': 'Organic compost bin',
    'glass': 'Glass recycling bin',
    'battery': 'Hazardous waste bin (battery disposal point)'
}

cap = cv2.VideoCapture(0)  # 0 for webcam, or give video path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_id = np.argmax(pred)
    label = class_names[class_id]

    cv2.putText(frame, f"Detected: {label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Waste Classification', frame)

    # Voice feedback
    speak(f"{label} detected. Please put it in the {bin_dict[label]}")

    if cv2.waitKey(20) & 0xFF == ord('q'):  # wait 2 sec, press q to quit
        break

cap.release()
cv2.destroyAllWindows()