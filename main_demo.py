from roboflow import Roboflow
import cv2
import numpy as np
import os
from ultralytics import YOLO
import pyttsx3
import time
import mysql.connector
import requests
from datetime import datetime, timedelta

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Establish database connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Nithin1234#",
    database="object_detection"
)
cursor = conn.cursor()

# API key and project settings
api_key = "5r6czWGmYRR4Kpqz1dKX"
project_name = "outside2-4qglu"
version = "2"
model_path = "yolov8s.pt"
video_path = r"C:\Users\nithi\Downloads\Walking in Chennai (India).mp4"
focal_length = 840  # Adjust focal length accordingly
object_real_width = 6.3  # Average width of the object in cm

# Initialize Roboflow and YOLO models
rf = Roboflow(api_key=api_key)
project = rf.workspace().project(project_name)
model_rf = project.version(version).model
model_yolo = YOLO(model_path)

class ObjectDetection:
    def __init__(self, model_rf, model_yolo, video_path, focal_length, object_real_width):
        self.model_rf = model_rf
        self.model_yolo = model_yolo
        self.cap = cv2.VideoCapture(0 if video_path is None else video_path)
        self.detected_objects = {}
        self.announcement_threshold = 20
        self.output_dir = 'data'
        os.makedirs(self.output_dir, exist_ok=True)
        self.focal_length = focal_length
        self.object_real_width = object_real_width
        self.frame_count = 0
        self.frames_to_announce = 5

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Error: Failed to capture frame.")
        return frame

    def save_frame(self, frame, path):
        cv2.imwrite(path, frame)

    def predict_objects_rf(self, frame_path):
        return self.model_rf.predict(frame_path, confidence=40, overlap=30).json()

    def process_predictions_rf(self, predictions):
        detected_objects = [pred['class'] for pred in predictions['predictions']]
        return detected_objects

    def insert_object_into_db(self, class_name, confidence, direction, angle, distance):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = '''
        INSERT INTO detections (class_name, confidence, direction, angle, distance, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s)
        '''
        values = (class_name, confidence, direction, angle, distance, timestamp)
        cursor.execute(query, values)
        conn.commit()

    def insert_location_into_db(self, latitude, longitude):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = '''
        INSERT INTO location (latitude, longitude, timestamp)
        VALUES (%s, %s, %s)
        '''
        values = (latitude, longitude, timestamp)
        cursor.execute(query, values)
        conn.commit()

    def get_center(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2

    def classify_direction(self, center_x, img_width):
        if center_x < img_width / 3:
            return "Left"
        elif center_x < 2 * img_width / 3:
            return "Center"
        else:
            return "Right"

    def calculate_angle(self, center_x, img_width):
        dx = center_x - img_width / 2
        return np.degrees(np.arctan(dx / self.focal_length))

    def estimate_distance(self, width_pixels):
        if width_pixels == 0:
            return float('inf')
        return (self.object_real_width * self.focal_length) / width_pixels

    def get_ip_location(self):
        try:
            response_ip = requests.get('https://api.ipify.org?format=json')
            current_ip = response_ip.json()['ip']
            response_location = requests.get(f'https://ipinfo.io/{current_ip}/json')
            location_data = response_location.json()
            latitude, longitude = location_data['loc'].split(',')
            return latitude, longitude
        except Exception as e:
            print(f"Error retrieving location: {e}")
            return None, None

    def announce_object(self, detected_objects):
        if detected_objects:
            object_counts = {}
            closest_objects = {}

            for obj in detected_objects:
                class_name = obj['class_name']
                distance = obj['distance']
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
                closest_objects[class_name] = min(closest_objects.get(class_name, float('inf')), distance)

            announcement = "There are "
            object_phrases = [f"{count} {cls}{'s' if count > 1 else ''}" for cls, count in object_counts.items()]
            announcement += ", ".join(object_phrases) + " in front of you."

            # Announce distance for specific objects
            for vehicle in ['motorcycle', 'bike', 'car']:
                if vehicle in closest_objects:
                    announcement += f" The closest {vehicle} is {closest_objects[vehicle]:.2f} meters away."

            print(f"Announcing: {announcement}")
            engine.say(announcement)
            engine.runAndWait()

    def run_detection(self):
        last_time = datetime.now()

        while self.cap.isOpened():
            frame = self.capture_frame()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            frame_path = f"data/live_frame_{timestamp}.jpg"
            self.save_frame(frame, frame_path)
            predictions_rf = self.predict_objects_rf(frame_path)

            if predictions_rf['predictions']:
                detected_objects_rf = self.process_predictions_rf(predictions_rf)
                print(f"Roboflow detected objects: {', '.join(detected_objects_rf)}")

            # YOLO predictions
            results_yolo = self.model_yolo(frame)
            detected_in_frame_yolo = []
            img_width = frame.shape[1]
            for result in results_yolo:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    width_pixels = x2 - x1
                    center_x = self.get_center((x1, y1, x2, y2))
                    direction = self.classify_direction(center_x, img_width)
                    angle = self.calculate_angle(center_x, img_width)
                    class_id = int(box.cls[0].tolist())
                    class_name = self.model_yolo.names[class_id]
                    confidence = box.conf[0].tolist()
                    distance = self.estimate_distance(width_pixels)

                    print(f"Class: {class_name}, Confidence: {confidence:.2f}, Direction: {direction}, Angle: {angle:.2f}°, Distance: {distance:.2f} cm")

                    # Insert object detection data into the database
                    self.insert_object_into_db(class_name, confidence, direction, angle, distance)

                    # Add detected object for announcement
                    detected_in_frame_yolo.append({"class_name": class_name, "distance": distance})

            # Announce objects every 8 frames
            self.frame_count += 1
            if self.frame_count % self.frames_to_announce == 0 and detected_in_frame_yolo:
                self.announce_object(detected_in_frame_yolo)

            # Get location data every 5 minutes
            current_time = datetime.now()
            if current_time - last_time >= timedelta(minutes=5):
                latitude, longitude = self.get_ip_location()
                if latitude and longitude:
                    self.insert_location_into_db(latitude, longitude)
                last_time = current_time

            # Show detection results
            annotated_frame = results_yolo[0].plot()
            cv2.imshow("Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetection(model_rf, model_yolo, video_path, focal_length, object_real_width)
    detector.run_detection()
