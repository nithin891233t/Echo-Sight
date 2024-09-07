import mysql.connector

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Nithin1234#",  # Use your MySQL password
    database="object_detection"
)
cursor = conn.cursor()

# Function to insert data into MySQL database
def insert_into_db(cursor, conn, class_name, confidence, direction, angle, distance):
    query = '''
    INSERT INTO detections (class_name, confidence, direction, angle, distance)
    VALUES (%s, %s, %s, %s, %s)
    '''
    values = (class_name, confidence, direction, angle, distance)
    cursor.execute(query, values)
    conn.commit()

# Example of inserting data
insert_into_db(cursor, conn, "car", 0.94, "Right", 12.78, 25.57)

# Close connection
cursor.close()
conn.close()
