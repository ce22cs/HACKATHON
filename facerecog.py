choose file with chatbot                                                                                                                                                                    # Install required libraries
!pip install deepface

# Import necessary libraries
from google.colab import files
import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
import random

# Emotion-based suggestions
suggestions = {
    "happy": ["Keep spreading the joy!", "Listen to your favorite upbeat song!", "Celebrate your happiness with a friend!"],
    "sad": ["Try listening to uplifting music.", "Go for a walk in nature.", "Watch a funny movie or talk to a friend."],
    "angry": ["Take deep breaths and try meditation.", "Go for a short walk.", "Listen to calming music to relax."],
    "surprise": ["Enjoy the moment!", "Share your excitement with someone!", "Capture this memory with a picture!"],
    "fear": ["Try deep breathing exercises.", "Write down your thoughts to understand them better.", "Listen to relaxing sounds."],
    "disgust": ["Try shifting your focus to something positive.", "Take a moment to reflect.", "Engage in a hobby you love."],
    "neutral": ["Take a break and do something you enjoy.", "Try learning something new.", "Relax and enjoy the present moment."]
}

# Ask user to upload an image
print("Please upload an image for emotion detection:")
uploaded = files.upload()  # Opens a file picker

# Get the uploaded file name
image_path = list(uploaded.keys())[0]

# Load the uploaded image
img = cv2.imread(image_path)

# Analyze emotion using DeepFace
try:
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    detected_emotion = result[0]['dominant_emotion']
    print(f"\nDetected Emotion: {detected_emotion}")

    # Display image with emotion text
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"Emotion: {detected_emotion}")
    plt.show()

    # Provide chatbot response and suggestions
    print("\nChatbot: You seem to be feeling", detected_emotion)
    suggestion = random.choice(suggestions.get(detected_emotion, ["Take care!"]))
    print(f"Chatbot Suggestion: {suggestion}")

except Exception as e:
    print(f"Error: {e}")