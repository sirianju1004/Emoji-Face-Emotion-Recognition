import cv2
from deepface import DeepFace
import tkinter as tk
from PIL import Image, ImageTk

# Load emoji images for detected emotions
emojis = {
    'angry': "emojis/Angry.png",
    'disgust': "emojis/Disgust.png",
    'fear': "emojis/Fear.png",
    'happy': "emojis/Happy.png",
    'sad': "emojis/Sad.png",
    'surprise': "emojis/Surprise.png",
    'neutral': "emojis/Neutral.png"
}

# OpenCV Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start Video Capture
cap = cv2.VideoCapture(0)

# Create Tkinter Window
root = tk.Tk()
root.title("😊 Emoji Face Emotion Recognition 😊")

# Set Window Size and Background Color
root.geometry("900x700")
root.configure(bg="#ffcccc")  # Light pink background

# Main Frame for Center White Box
main_frame = tk.Frame(root, bg="white", bd=5, relief="solid")
main_frame.place(relx=0.5, rely=0.5, anchor="center", width=550, height=650)

# Title Label
title_label = tk.Label(main_frame, text="😊 Emoji Face Emotion Recognition 😊", font=("Segoe UI Emoji", 18, "bold"),
                       bg="white", fg="green")
title_label.pack(pady=10)

# Video Frame
video_frame = tk.Label(main_frame, bg="white", relief="solid", bd=3)
video_frame.pack(pady=10)

# Emotion Text Label
emotion_label = tk.Label(main_frame, text="", font=("Arial", 16, "bold"), fg="black", bg="white")
emotion_label.pack(pady=10)

# Emoji Image Label
emoji_label = tk.Label(main_frame, bg="white")
emoji_label.pack(pady=10)

# Footer Text with Name and Heart Emoji
footer_label = tk.Label(root, text="Made by Ankita Meshram ❤️", font=("Arial", 12, "bold"), fg="black", bg="#ffcccc")
footer_label.place(relx=0.5, rely=0.98, anchor="center")

# Function to Process Video Frame
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        try:
            # Emotion Detection
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']

            # Update Emotion Label
            emotion_label.config(text=f"Emotion Detected: {emotion.upper()}", fg="black")

            # Load and Display Emoji for detected emotion
            emoji_path = emojis.get(emotion, None)
            if emoji_path:
                img = Image.open(emoji_path)
                img = img.resize((100, 100))  # Resize emoji
                img = ImageTk.PhotoImage(img)
                emoji_label.config(image=img)
                emoji_label.image = img

        except Exception as e:
            print("Error:", e)

    # Convert Frame for Display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (400, 300))
    img = ImageTk.PhotoImage(Image.fromarray(frame))

    # Update Video Frame
    video_frame.config(image=img)
    video_frame.image = img
    root.after(10, update_frame)


# Start Video Stream
update_frame()

# Run Tkinter GUI
root.mainloop()

# Release Resources
cap.release()
cv2.destroyAllWindows()
