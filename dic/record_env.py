import cv2
import datetime

def take_photo():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    ret, frame = camera.read()

    if ret:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename = f"photo_{timestamp}.jpg"

        cv2.imwrite(filename, frame)

        print(f"Photo saved as {filename}")
    else:
        print("Error: Could not capture photo.")

    # Release the camera
    camera.release()
    cv2.destroyAllWindows()


def record_video():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"video_{timestamp}.avi"

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    print("Recording video. Press 'q' to stop...")

    while True:
        ret, frame = camera.read()

        if ret:
            out.write(frame)

            cv2.imshow('Recording...', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    print(f"Video saved as {filename}")
    camera.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    while True:
        print("Choose an option:")
        print("1. Take a Photo")
        print("2. Record a Video")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            take_photo()
        elif choice == '2':
            record_video()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
