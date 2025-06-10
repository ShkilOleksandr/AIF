import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2
import numpy as np

# -------------------------
#  CONFIGURATION
# -------------------------
RAF_MODEL_FILE = "raf_emotion_resnet18.pth"
FER_MODEL_FILE = "fer_emotion_cnn.pth"
EMOTIONS_RAF = ['surprise','fear','disgust','happy','sad','anger','neutral']
EMOTIONS_FER = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU

# -------------------------
#  MODEL LOADING
# -------------------------
def load_raf_model():
    if not os.path.exists(RAF_MODEL_FILE):
        raise FileNotFoundError(f"RAF model not found: {RAF_MODEL_FILE}")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(EMOTIONS_RAF))
    model.load_state_dict(torch.load(RAF_MODEL_FILE, map_location='cpu'))
    model.eval()
    return model


def load_fer_model():
    if not os.path.exists(FER_MODEL_FILE):
        raise FileNotFoundError(f"FER model not found: {FER_MODEL_FILE}")
    # assume FER model uses same ResNet-18 architecture
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(EMOTIONS_FER))
    model.load_state_dict(torch.load(FER_MODEL_FILE, map_location='cpu'))
    model.eval()
    return model

raf_model = None
fer_model = None

# -------------------------
#  INFERENCE
# -------------------------
def analyze_raf(img_path):
    global raf_model
    if raf_model is None:
        raf_model = load_raf_model()
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img = Image.open(img_path).convert("RGB")
    inp = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = raf_model(inp)
        probs = torch.nn.functional.softmax(logits[0], dim=0).numpy()
    result = {EMOTIONS_RAF[i]: float(probs[i]) for i in range(len(EMOTIONS_RAF))}
    dominant = EMOTIONS_RAF[int(np.argmax(probs))]
    return dominant, result


def analyze_fer(img_path):
    global fer_model
    if fer_model is None:
        fer_model = load_fer_model()
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img = Image.open(img_path).convert("RGB")
    inp = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = fer_model(inp)
        probs = torch.nn.functional.softmax(logits[0], dim=0).numpy()
    result = {EMOTIONS_FER[i]: float(probs[i]) for i in range(len(EMOTIONS_FER))}
    dominant = EMOTIONS_FER[int(np.argmax(probs))]
    return dominant, result


def analyze_deepseak(img_path):
    res = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False)[0]
    return res['dominant_emotion'], res['emotion']


def analyze_emotion(img_path, method):
    if method == "RAF":
        return analyze_raf(img_path)
    elif method == "FER":
        return analyze_fer(img_path)
    else:  # DeepSeak
        return analyze_deepseak(img_path)

# -------------------------
#  PLOT RESULTS
# -------------------------
def plot_emotions(emotions, dominant, img_path):
    plt.figure(figsize=(6,4))
    plt.bar(emotions.keys(), emotions.values())
    plt.title(f"Dominant: {dominant}")
    plt.ylabel("Confidence")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    img = Image.open(img_path)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title("Input Image")
    plt.show()

# -------------------------
#  TKINTER GUI
# -------------------------
class EmotionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emotion Recognition App")
        self.geometry("500x480")

        self.model_choice = tk.StringVar(value="DeepFace")
        self.image_path = None

        ttk.Label(self, text="Choose Model:").pack(pady=5)
        ttk.Radiobutton(self, text="RAF-DB",       variable=self.model_choice, value="RAF").pack()
        ttk.Radiobutton(self, text="FER2013",       variable=self.model_choice, value="FER").pack()
        ttk.Radiobutton(self, text="DeepFace",  variable=self.model_choice, value="DeepFace").pack()

        ttk.Button(self, text="Load Image",        command=self.load_image).pack(pady=5)
        ttk.Button(self, text="Capture from Webcam", command=self.capture_image).pack(pady=5)
        ttk.Button(self, text="Analyze Emotion",    command=self.run_analysis).pack(pady=10)

        self.preview_label = ttk.Label(self, text="No image selected")
        self.preview_label.pack()

        # separator line at bottom
        sep = ttk.Separator(self, orient='horizontal')
        sep.pack(fill='x', pady=(20,5))

        self.credit_label = ttk.Label(
            self,
            text="by Shkil, Nosal, Bondarenko, Fishchuk",
            font=("Segoe UI", 8),
            foreground="gray"
        )
        self.credit_label.pack(side="bottom", pady=(0,5))

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
            if key == 27:
                break
            elif key == 32:
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
        self.preview_label.image = tk_img

    def run_analysis(self):
        if not self.image_path:
            messagebox.showerror("No Image", "Please load or capture an image.")
            return

        try:
            dom, emotions = analyze_emotion(self.image_path, self.model_choice.get())
            plot_emotions(emotions, dom, self.image_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    EmotionApp().mainloop()