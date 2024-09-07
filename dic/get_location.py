import requests
from datetime import datetime, timedelta


def get_ip_location():
    try:
        # Get the public IP address
        response_ip = requests.get('https://api.ipify.org?format=json')
        current_ip = response_ip.json()['ip']

        # Get location data based on the IP address
        response_location = requests.get(f'https://ipinfo.io/{current_ip}/json')
        location_data = response_location.json()
        latitude, longitude = location_data['loc'].split(',')

        return latitude, longitude
    except Exception as e:
        print(f"Error retrieving location: {e}")
        return None, None


def save_location_to_file(latitude, longitude, filename="location_log.txt"):
    try:
        with open(filename, "a") as file:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"Timestamp: {current_time}\n")
            file.write(f"Latitude: {latitude}\n")
            file.write(f"Longitude: {longitude}\n")
            file.write("\n")
        print(f"Location saved at {current_time}")
    except Exception as e:
        print(f"Error saving location to file: {e}")


if __name__ == "__main__":
    # Store the first IP location immediately
    latitude, longitude = get_ip_location()
    if latitude and longitude:
        save_location_to_file(latitude, longitude)
    else:
        print("Failed to retrieve location.")

    # Set the last_time to the current time
    last_time = datetime.now()

    while True:
        current_time = datetime.now()

        # Check if 5 minutes (300 seconds) have passed
        if current_time - last_time >= timedelta(minutes=5):
            # Retrieve the IP-based location
            latitude, longitude = get_ip_location()

            if latitude and longitude:
                # Save the location to a file with timestamp
                save_location_to_file(latitude, longitude)
                last_time = current_time
            else:
                print("Failed to retrieve location.")
