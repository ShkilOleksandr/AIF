import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

# # Force CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

CUSTOM_MODEL_FILENAME = "emotion_cnn.pth"
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
custom_model = None

def load_custom_model():
    model_path = os.path.join(os.path.dirname(__file__), CUSTOM_MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Custom model not found at: {model_path}")

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model

def analyze_with_custom_model(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = custom_model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        result = {emotion_labels[i]: float(probs[i]) for i in range(len(emotion_labels))}
        dominant = emotion_labels[torch.argmax(probs)]
        return dominant, result

def analyze_emotion(image_path, model_choice):
    if model_choice == "DeepFace":
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)[0]
        return result['dominant_emotion'], result['emotion']
    elif model_choice == "Custom":
        return analyze_with_custom_model(image_path)

def plot_emotions(emotions, dominant_emotion, image_path):
    plt.figure(figsize=(6, 4))
    plt.bar(emotions.keys(), emotions.values())
    plt.title(f"Dominant Emotion: {dominant_emotion}")
    plt.ylabel("Confidence")
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = Image.open(image_path)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title("Input Image")
    plt.show()

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition App")
        self.root.geometry("500x400")

        self.model_choice = tk.StringVar(value="DeepFace")
        self.image_path = None

        ttk.Label(root, text="Choose Emotion Model").pack(pady=5)
        ttk.Radiobutton(root, text="DeepFace", variable=self.model_choice, value="DeepFace").pack()
        ttk.Radiobutton(root, text="Custom (.pth)", variable=self.model_choice, value="Custom").pack()

        ttk.Button(root, text="Load Image", command=self.load_image).pack(pady=5)
        ttk.Button(root, text="Capture from Webcam", command=self.capture_image).pack(pady=5)
        ttk.Button(root, text="Analyze Emotion", command=self.run_analysis).pack(pady=10)

        self.preview_label = ttk.Label(root, text="No image selected")
        self.preview_label.pack()

        self.credit_label = ttk.Label(
        root,
        text="by Shkil, Nosal, Bondarenko, Fishchuk",
        font=("Segoe UI", 8),
        foreground="gray"
        )
        self.credit_label.pack(side="bottom", pady=(10,5))

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if path:
            self.image_path = path
            self.show_image_preview(path)

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Webcam Error", "Webcam not accessible.")
            return

        messagebox.showinfo("Webcam", "Press SPACE to capture. ESC to cancel.")
        
        cv2.namedWindow("Webcam")

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Capture Failed", "Could not read from webcam.")
                break

            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key
                break
            elif key == 32:  # SPACE key
                self.image_path = "captured.jpg"
                cv2.imwrite(self.image_path, frame)
                break

        cap.release()
        cv2.destroyAllWindows()

        if self.image_path:
            self.show_image_preview(self.image_path)


    def show_image_preview(self, path):
        img = Image.open(path)
        img.thumbnail((300, 300))
        tk_img = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=tk_img, text="")
        self.preview_label.image = tk_img  # keep a reference!

    def run_analysis(self):
        if not self.image_path:
            messagebox.showerror("No Image", "Please load or capture an image.")
            return

        if self.model_choice.get() == "Custom":
            global custom_model
            if custom_model is None:
                try:
                    custom_model = load_custom_model()
                except Exception as e:
                    messagebox.showerror("Model Load Error", str(e))
                    return

        try:
            dominant, emotions = analyze_emotion(self.image_path, self.model_choice.get())
            plot_emotions(emotions, dominant, self.image_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
