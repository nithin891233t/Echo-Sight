from roboflow import Roboflow
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import pyttsx3
import mysql.connector
import requests
from datetime import datetime, timedelta

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Establish a connection to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password=" ",  # Use your MySQL password
    database="object_detection"
)
cursor = conn.cursor()

# API key and project settings
api_key = "API_KEY"
project_name = "PROJECT_NAME"
version = "0"
model_path = "yolov8s.pt"
video_path =  "ADD PATH"       #r"C:\Users\nithi\Downloads\Walking in Chennai (India).mp4"
focal_length = 840                            # Replace with actual focal length of your camera
object_real_width = 6.3                       # Average width of object in cm

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
        self.detector = FaceMeshDetector(maxFaces=1)
        self.focal_length = focal_length
        self.object_real_width = object_real_width

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

    def announce_object(self, object_names):
        current_time = time.time()
        for object_info in object_names:
            class_name = object_info["class_name"]
            distance = object_info["distance"]
            last_announcement = self.detected_objects.get(class_name, 0)
            if current_time - last_announcement > self.announcement_threshold:
                print(f"There is a {class_name} in front of you at {distance:.2f} meters.")
                self.detected_objects[class_name] = current_time
                engine.say(f"There is a {class_name} in front of you at {distance:.2f} meters.")
                engine.runAndWait()

    def insert_object_into_db(self, cursor, conn, class_name, confidence, direction, angle, distance):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = '''
        INSERT INTO detections (class_name, confidence, direction, angle, distance, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s)
        '''
        values = (class_name, confidence, direction, angle, distance, timestamp)
        cursor.execute(query, values)
        conn.commit()

    def insert_location_into_db(self, cursor, conn, latitude, longitude):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = '''
        INSERT INTO location (latitude, longitude)
        VALUES (%s, %s,)
        '''
        values = (latitude, longitude)
        cursor.execute(query, values)
        conn.commit()

    def get_center(self, box):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        return center_x

    def classify_direction(self, center_x, img_width):
        if center_x < img_width / 3:
            return "Left"
        elif center_x < 2 * img_width / 3:
            return "Center (Straight)"
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

    def run_detection(self):
        last_time = datetime.now()
        while self.cap.isOpened():
            frame = self.capture_frame()

            frame_path = "live_frame.jpg"
            self.save_frame(frame, frame_path)
            predictions_rf = self.predict_objects_rf(frame_path)
            if predictions_rf['predictions']:
                detected_objects_rf = self.process_predictions_rf(predictions_rf)
                print(f"Roboflow detected objects: {', '.join(detected_objects_rf)}")

            results_yolo = self.model_yolo(frame)
            detected_in_frame_yolo = []
            img_width = frame.shape[1]
            for result in results_yolo:
                boxes = result.boxes
                for box in boxes:
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
                    self.insert_object_into_db(cursor, conn, class_name, confidence, direction, angle, distance)

                    # Adding the detected object with distance information
                    detected_in_frame_yolo.append({"class_name": class_name, "distance": distance})

            if detected_in_frame_yolo:
                self.announce_object(detected_in_frame_yolo)

            # Get location data every 5 minutes
            current_time = datetime.now()
            if current_time - last_time >= timedelta(minutes=2):
                latitude, longitude = self.get_ip_location()
                if latitude and longitude:
                    # Insert location data into the database
                    self.insert_location_into_db(cursor, conn, latitude, longitude)
                last_time = current_time

            # Face distance estimation
            img, faces = self.detector.findFaceMesh(frame, draw=False)
            if faces:
                face = faces[0]
                point_left = face[145]
                point_right = face[374]
                distance = self.estimate_distance(self.detector.findDistance(point_left, point_right)[0])
                cvzone.putTextRect(img, f'Depth: {int(distance)}cm', (face[10][0] - 100, face[10][1] - 50), scale=2)

            # Save annotated frame
            annotated_frame = results_yolo[0].plot()
            current_time = time.time()
            output_path = os.path.join(self.output_dir, f'output_frame_{int(current_time)}.jpg')
            cv2.imwrite(output_path, annotated_frame)

            cv2.imshow("Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetection(model_rf, model_yolo, video_path, focal_length, object_real_width)
    detector.run_detection()
