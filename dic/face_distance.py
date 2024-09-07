import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

class FaceDetection:
    def __init__(self, focal_length=840, object_real_width=6.3):
        self.detector = FaceMeshDetector(maxFaces=1)
        self.focal_length = focal_length
        self.object_real_width = object_real_width

    def detect_faces(self, frame):
        return self.detector.findFaceMesh(frame, draw=False)

    def estimate_distance(self, point_left, point_right):
        w, _ = self.detector.findDistance(point_left, point_right)
        distance = (self.object_real_width * self.focal_length) / w
        return distance
