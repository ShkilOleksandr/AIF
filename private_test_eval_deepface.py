import pandas as pd
from deepface import DeepFace
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

# Load FER2013 dataset
df = pd.read_csv("fer2013.csv")
private_test_df = df[df['Usage'] == 'PrivateTest']

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def pixels_to_image(pixels):
    arr = np.fromstring(pixels, sep=' ', dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(arr)
    img = img.convert("RGB")
    return img

y_true = []
y_pred = []

print("Evaluating DeepFace on FER2013 PrivateTest set...")

for idx, row in tqdm(private_test_df.iterrows(), total=len(private_test_df)):
    img = pixels_to_image(row['pixels'])
    img_path = f"temp_{idx}.jpg"
    img.save(img_path)
    try:
        result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)[0]
        pred_emotion = result['dominant_emotion'].capitalize()
        if pred_emotion not in emotion_labels:
            pred_emotion = 'Neutral'  # fallback for unknown
    except Exception:
        pred_emotion = 'Neutral'
    y_pred.append(emotion_labels.index(pred_emotion))
    y_true.append(row['emotion'])

# Clean up temp images
# does not work
for idx in range(len(private_test_df)):
    img_path = f"temp_{idx}.jpg"
    if os.path.exists(img_path):
        os.remove(img_path)

print(classification_report(y_true, y_pred, target_names=emotion_labels))