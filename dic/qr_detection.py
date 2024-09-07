import numpy as np
import cv2
from pyzbar.pyzbar import decode


def qr_code_scanner():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    scanned_qr_codes = set()
    urls = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        decoded_objects = decode(frame)

        for obj in decoded_objects:

            points = obj.polygon
            if len(points) == 4:
                pts = [(point.x, point.y) for point in points]

                cv2.polylines(frame, [np.array(pts, np.int32)], True, (0, 255, 0), 2)

            qr_data = obj.data.decode("utf-8")

            if qr_data not in scanned_qr_codes:

                print(f"QR Code detected: {qr_data}")

                if qr_data.startswith("http://") or qr_data.startswith("https://"):
                    urls.append(qr_data)
                    with open("detected_urls.txt", "w") as file:
                        file.write(urls + "\n")
                scanned_qr_codes.add(qr_data)

        cv2.imshow("QR Code Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    qr_code_scanner()
