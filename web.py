import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response
import threading
import time

# Load your trained model
model = tf.keras.models.load_model("face_detect_model.h5")

# Shared variables for threading
frame_to_predict = None
last_label = "Loading..."
last_color = (255, 255, 0)
lock = threading.Lock()

# Prediction thread
def predict_loop():
    global frame_to_predict, last_label, last_color
    while True:
        if frame_to_predict is not None:
            img_resized = cv2.resize(frame_to_predict, (224, 224))
            y_pred = model.predict(img_resized.reshape(1, 224, 224, 3), verbose=0)[0][0]

            lock.acquire()
            if y_pred < 0.1:  # adjust threshold
                last_label = "with_mask"
                last_color = (0, 255, 0)
            else:
                last_label = "without_mask"
                last_color = (0, 0, 255)
            lock.release()
            frame_to_predict = None
        else:
            time.sleep(0.01)  # small sleep to avoid CPU overuse

# Start prediction thread
threading.Thread(target=predict_loop, daemon=True).start()

# Function to draw label
def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)
    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2
    cv2.rectangle(img, pos, (end_x, end_y), color=bg_color, thickness=-1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

# ---------------- Flask App ----------------
app = Flask(__name__)

# Video generator for streaming
def generate_frames():
    global frame_to_predict
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not detected")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Send a copy to prediction thread if free
        if frame_to_predict is None:
            frame_to_predict = frame.copy()

        # Draw last prediction
        lock.acquire()
        draw_label(frame, last_label, (30, 30), last_color)
        lock.release()

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Face Mask Detection</title>
      <style>
        body { font-family: Arial; text-align: center; background: #f4f4f4; margin: 0; padding: 0; }
        header { background: #2c3e50; color: #fff; padding: 20px; }
        h1 { margin: 0; }
        .video-container { margin-top: 20px; }
        img { border: 5px solid #2c3e50; border-radius: 10px; box-shadow: 0px 4px 15px rgba(0,0,0,0.2); }
        footer { margin-top: 20px; padding: 10px; font-size: 14px; color: #555; }
      </style>
    </head>
    <body>
      <header><h1>Live Face Mask Detection</h1></header>
      <div class="video-container">
        <img src="/video_feed" width="640" height="480" alt="Live Video">
      </div>
      <footer><p>Flask + OpenCV + TensorFlow</p></footer>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
