import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from collections import deque
import threading
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EmotionResNet18(nn.Module):
    def __init__(self, num_classes=6):
        super(EmotionResNet18, self).__init__()
        # Use ResNet18 directly without any wrapper
        # Copy all ResNet18 layers directly to this class
        resnet = models.resnet18(pretrained=False)
        
        # Copy all layers directly
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(resnet.fc.in_features, 6)  # 6 classes as trained
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Détecteur d'Émotions - ResNet18")
        self.root.geometry("1200x700")
        self.root.configure(bg='#f5f7fa')
        
        self.model_path = 'emotion_resnet18_best.pth'
        
        self.emotion_image_paths = {
            'angry': 'images/colere.png',
            'disgust': 'images/degout.png',
            'fear': 'images/peur.png',
            'happy': 'images/joie.png',
            'sad': 'images/tristesse.png',
            'neutral': 'images/neutre.png'
        }
        
        # Variables
        self.is_running = False
        self.cap = None
        self.current_emotion = "neutral"
        
        # Session recording (video only)
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        
        # 6 emotions for ResNet18 (as trained)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
        
        self.emotion_colors = {
            'angry': '#ef4444',
            'disgust': '#8b5cf6',
            'fear': "#ffaf02",
            'happy': "#eeff00",
            'sad': '#3b82f6',
            'neutral': '#6b7280'
        }
        self.emotion_history = deque(maxlen=50)
        
        self.emotion_images = {}
        
        # Detect device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the PyTorch model
        self.model = None
        self.load_model()
       
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            messagebox.showerror("Erreur", "Impossible de charger le détecteur de visage")
        
        self.load_emotion_images()
        
        # Create sessions directory
        self.sessions_dir = "sessions"
        if not os.path.exists(self.sessions_dir):
            os.makedirs(self.sessions_dir)
        
        self.setup_ui()
    
    def load_model(self):
        """Load the trained ResNet18 model"""
        try:
            if not os.path.exists(self.model_path):
                messagebox.showwarning(
                    "Attention", 
                    f"Modèle ResNet18 introuvable: {self.model_path}\n\nLe mode simulation sera utilisé."
                )
                print("Mode simulation activé - aucun modèle chargé")
                return
            
            self.model = EmotionResNet18(num_classes=6)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Modèle ResNet18 chargé avec succès depuis {self.model_path}")
            
        except Exception as e:
            messagebox.showerror(
                "Erreur", 
                f"Erreur lors du chargement du modèle ResNet18:\n{str(e)}\n\nLe mode simulation sera utilisé."
            )
            print(f"Erreur de chargement: {str(e)}")
            self.model = None
        
    def load_emotion_images(self):
        """Load all emotion images from the specified paths"""
        missing_images = []
        
        for emotion, path in self.emotion_image_paths.items():
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    img.thumbnail((250, 400), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.emotion_images[emotion] = photo
                except Exception as e:
                    missing_images.append(f"{emotion}: {str(e)}")
            else:
                missing_images.append(f"{emotion}: fichier introuvable ({path})")
        
        if missing_images:
            error_msg = "Images manquantes ou erreurs:\n\n" + "\n".join(missing_images)
            messagebox.showwarning("Attention", error_msg)
        
    def setup_ui(self):
        # Set window to fullscreen
        self.root.attributes('-fullscreen', True)
        
        # Optional: Add escape key binding to exit fullscreen
        self.root.bind('<Escape>', lambda e: self.root.attributes('-fullscreen', False))
        header_frame = tk.Frame(self.root, bg='#ffffff', height=70)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="Détecteur d'Émotions - ResNet18",
            font=('Segoe UI', 24, 'bold'),
            bg='#ffffff',
            fg='#1f2937'
        )
        title_label.pack(pady=20)
        
        # Frame principal avec padding
        main_frame = tk.Frame(self.root, bg='#f5f7fa')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # Frame gauche - Vidéo (plus large)
        left_frame = tk.Frame(main_frame, bg='#ffffff', relief=tk.FLAT, bd=0)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        self.video_canvas = tk.Label(left_frame, bg='#000000')
        self.video_canvas.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)
        
        # Frame droit - Informations compactes
        right_frame = tk.Frame(main_frame, bg='#ffffff', relief=tk.FLAT, bd=0, width=320)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_frame.pack_propagate(False)
        
        # Émotion actuelle 
        emotion_container = tk.Frame(right_frame, bg='#ffffff')
        emotion_container.pack(pady=25, padx=20, fill=tk.X)
        
        tk.Label(
            emotion_container,
            text="Émotion Détectée",
            font=('Segoe UI', 11),
            bg='#ffffff',
            fg='#6b7280'
        ).pack()
        
        self.emotion_label = tk.Label(
            emotion_container,
            text="neutral",
            font=('Segoe UI', 36, 'bold'),
            bg='#ffffff',
            fg='#6b7280'
        )
        self.emotion_label.pack(pady=5)
        
        image_container = tk.Frame(right_frame, bg='#f9fafb', relief=tk.FLAT, bd=0)
        image_container.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.emotion_image_canvas = tk.Label(
            image_container, 
            bg='#f9fafb',
            text="",
            font=('Segoe UI', 10),
            fg='#9ca3af'
        )
        self.emotion_image_canvas.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)
        
        if 'neutral' in self.emotion_images:
            self.display_emotion_image('neutral')
        
        prob_frame = tk.Frame(right_frame, bg='#ffffff')
        prob_frame.pack(pady=20, padx=20, fill=tk.X)
        
        tk.Label(
            prob_frame,
            text="Probabilités",
            font=('Segoe UI', 11, 'bold'),
            bg='#ffffff',
            fg='#1f2937'
        ).pack(pady=(0, 15), anchor='w')
        
        self.prob_labels = {}
        self.prob_bars = {}
        
        for emotion in self.emotions:
            emotion_row = tk.Frame(prob_frame, bg='#ffffff')
            emotion_row.pack(fill=tk.X, pady=5)
            
            label = tk.Label(
                emotion_row,
                text=emotion,
                font=('Segoe UI', 9),
                bg='#ffffff',
                fg='#374151',
                width=9,
                anchor='w'
            )
            label.pack(side=tk.LEFT)
            
            bar_bg = tk.Frame(emotion_row, bg='#e5e7eb', height=8)
            bar_bg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
            
            bar = tk.Frame(bar_bg, bg=self.emotion_colors[emotion], height=8)
            bar.place(x=0, y=0, relwidth=0, relheight=1)
            self.prob_bars[emotion] = bar
            
            prob_label = tk.Label(
                emotion_row,
                text="0%",
                font=('Segoe UI', 9, 'bold'),
                bg='#ffffff',
                fg='#6b7280',
                width=4
            )
            prob_label.pack(side=tk.RIGHT)
            self.prob_labels[emotion] = prob_label
        
        control_frame = tk.Frame(self.root, bg='#ffffff', height=80)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM)
        control_frame.pack_propagate(False)
        
        tk.Frame(control_frame, bg='#e5e7eb', height=1).pack(fill=tk.X)
        
        button_container = tk.Frame(control_frame, bg='#ffffff')
        button_container.pack(expand=True)
        
        self.start_button = tk.Button(
            button_container,
            text="Démarrer",
            command=self.start_detection,
            font=('Segoe UI', 11, 'bold'),
            bg='#10b981',
            fg='white',
            width=14,
            height=1,
            relief=tk.FLAT,
            cursor='hand2',
            bd=0
        )
        self.start_button.pack(side=tk.LEFT, padx=8)
        
        self.stop_button = tk.Button(
            button_container,
            text="Arrêter",
            command=self.stop_detection,
            font=('Segoe UI', 11, 'bold'),
            bg='#ef4444',
            fg='white',
            width=14,
            height=1,
            relief=tk.FLAT,
            cursor='hand2',
            state=tk.DISABLED,
            bd=0
        )
        self.stop_button.pack(side=tk.LEFT, padx=8)
        
        # Recording button (video only)
        self.record_button = tk.Button(
            button_container,
            text="🔴 Enregistrer",
            command=self.toggle_recording,
            font=('Segoe UI', 11, 'bold'),
            bg='#dc2626',
            fg='white',
            width=14,
            height=1,
            relief=tk.FLAT,
            cursor='hand2',
            bd=0
        )
        self.record_button.pack(side=tk.LEFT, padx=8)
        
        # Model status indicator
        model_status = "ResNet18: Chargé" if self.model is not None else "ResNet18: Simulation"
        model_color = '#10b981' if self.model is not None else '#f59e0b'
        
        self.model_status_label = tk.Label(
            button_container,
            text=model_status,
            font=('Segoe UI', 9),
            bg='#ffffff',
            fg=model_color
        )
        self.model_status_label.pack(side=tk.LEFT, padx=15)
        
        self.status_label = tk.Label(
            button_container,
            text="Prêt",
            font=('Segoe UI', 10),
            bg='#ffffff',
            fg='#6b7280'
        )
        self.status_label.pack(side=tk.LEFT, padx=15)
    
    def display_emotion_image(self, emotion):
        """Display the image associated with the detected emotion"""
        if emotion in self.emotion_images:
            self.emotion_image_canvas.config(
                image=self.emotion_images[emotion],
                text='',
                bg='#f9fafb'
            )
            self.emotion_image_canvas.image = self.emotion_images[emotion]
        else:
            self.emotion_image_canvas.config(
                image='',
                text=f"Image non disponible",
                font=('Segoe UI', 10),
                fg='#9ca3af',
                bg='#f9fafb'
            )
        
    def preprocess_face(self, face_img):
        """Prétraiter l'image du visage pour le modèle ResNet18"""
        # Resize to model input size (48x48)
        face_img = cv2.resize(face_img, (48, 48))
        
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Normalize to [0, 1]
        face_img = face_img / 255.0
        
        # Convert to PyTorch tensor
        face_tensor = torch.from_numpy(face_img).float()
        
        # Add batch and channel dimensions: (1, 1, 48, 48)
        face_tensor = face_tensor.unsqueeze(0).unsqueeze(0)
        
        return face_tensor
    
    def predict_emotion(self, face_img):
        """Prédire l'émotion à partir d'une image de visage avec ResNet18"""
        
        # If model not loaded, use simulation
        if self.model is None:
            np.random.seed(int(time.time() * 1000) % 2**32)
            probs = np.random.dirichlet(np.ones(6) * 5)  # 6 emotions
            predictions = dict(zip(self.emotions, probs))
            emotion = max(predictions, key=predictions.get)
            return emotion, predictions
        
        try:
            # Preprocess the face image
            face_tensor = self.preprocess_face(face_img)
            face_tensor = face_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                probs = probabilities.cpu().numpy()[0]
            
            # Create predictions dictionary
            predictions = dict(zip(self.emotions, probs))
            
            # Get the emotion with highest probability
            emotion = max(predictions, key=predictions.get)
            
            return emotion, predictions
            
        except Exception as e:
            print(f"Erreur de prédiction: {str(e)}")
            predictions = {emotion: 1.0/6 for emotion in self.emotions}
            return 'neutral', predictions
    
    def update_probabilities(self, probabilities):
        """Mettre à jour l'affichage des probabilités"""
        for emotion, prob in probabilities.items():
            self.prob_bars[emotion].place(relwidth=prob)
            self.prob_labels[emotion].config(text=f"{int(prob * 100)}%")
    
    def start_detection(self):
        """Démarrer la détection"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Erreur", "Impossible d'accéder à la webcam")
                return
            
            self.is_running = True
            self.start_button.config(state=tk.DISABLED, bg='#9ca3af')
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="En cours...", fg='#10b981')
            
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du démarrage: {str(e)}")
    
    def stop_detection(self):
        """Arrêter la détection"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        self.start_button.config(state=tk.NORMAL, bg='#10b981')
        self.stop_button.config(state=tk.DISABLED, bg='#9ca3af')
        self.status_label.config(text="Arrêté", fg='#ef4444')
        self.video_canvas.config(image='')
        
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
    
    def detection_loop(self):
        """Boucle principale de détection"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
            )
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                emotion, probabilities = self.predict_emotion(face_roi)
                
                color = self.emotion_colors.get(emotion, '#ffffff')
                color_bgr = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 3)
                
                cv2.putText(
                    frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2
                )
                
                self.root.after(0, self.update_emotion_display, emotion, probabilities)
            
            # Record video if recording
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            
            self.root.after(0, self.update_video, photo)
            
            time.sleep(0.03)
    
    def toggle_recording(self):
        """Toggle recording on/off"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording video session"""
        if not self.is_running:
            messagebox.showwarning("Attention", "Veuillez d'abord démarrer la détection")
            return
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.recording_filename = f"{self.sessions_dir}/session_{timestamp}.mp4"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.recording_filename, 
            fourcc, 
            20.0, 
            (640, 480)
        )
        
        self.is_recording = True
        self.recording_start_time = time.time()
        
        self.record_button.config(
            text="⏹️ Arrêter",
            bg='#059669'
        )
        
        messagebox.showinfo("Enregistrement", f"Vidéo enregistrée: {self.recording_filename}")
    
    def stop_recording(self):
        """Stop recording session"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        
        self.record_button.config(
            text="🔴 Enregistrer",
            bg='#dc2626'
        )
        
        messagebox.showinfo("Enregistrement", "Enregistrement arrêté")
    
    def update_video(self, photo):
        """Mettre à jour l'affichage vidéo"""
        self.video_canvas.config(image=photo)
        self.video_canvas.image = photo
    
    def update_emotion_display(self, emotion, probabilities):
        """Mettre à jour l'affichage de l'émotion"""
        self.current_emotion = emotion
        self.emotion_label.config(
            text=emotion,
            fg=self.emotion_colors.get(emotion, '#6b7280')
        )
        self.update_probabilities(probabilities)
        self.display_emotion_image(emotion)


def main():
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    
    def on_closing():
        app.stop_detection()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
