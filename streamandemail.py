import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import random
import tempfile
import os
import base64
import re
import sqlite3
import email.mime.text
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from dotenv import load_dotenv
import gdown
# Set seed for reproducibility
seed = 1230
np.random.seed(seed)
random.seed(seed)

# Predefined coordinates for lines
paths = [
    {'id': 0, 'points': [(434, 123), (524, 126), (294, 645), (11, 610), (435, 123)]},
    {'id': 1, 'points': [(534, 128), (611, 131), (578, 715), (291, 712), (534, 128)]},
    {'id': 2, 'points': [(619, 132), (705, 133), (918, 717), (601, 719), (621, 133)]},
    {'id': 3, 'points': [(717, 135), (823, 134), (1272, 702), (948, 716), (720, 135)]}
]

url = "https://drive.google.com/uc?id=10qUkurq0X30k49dnUEmQbRGTuXkdAT4J&export=download"
output = "fine_tuned_modelv4.pt"

# Download the model
gdown.download(url, output, quiet=False)

# Load the YOLOv8 model
model = YOLO(output)
model.to('cpu')  # If you're using CPU
# Initialize DeepSORT tracker
tracker = DeepSort(max_age=4)

# Dictionary to log crossing information for objects
crossed_objects_log = {}

# Function to calculate centroid of a bounding box
def calculate_centroid(bbox):
    x1, y1, x2, y2 = bbox
    centroid_x = int((x1 + x2) / 2)
    centroid_y = int((y1 + y2) / 2)
    return centroid_x, centroid_y

# Function to calculate distance from a point (centroid) to a line segment (pt1-pt2)
def distance_from_line(pt1, pt2, centroid):
    line_magnitude = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    if line_magnitude == 0:
        return np.sqrt((centroid[0] - pt1[0]) ** 2 + (centroid[1] - pt1[1]) ** 2)

    u = ((centroid[0] - pt1[0]) * (pt2[0] - pt1[0]) + (centroid[1] - pt1[1]) * (pt2[1] - pt1[1])) / (line_magnitude ** 2)
    u = max(min(u, 1), 0)
    closest_point = (int(pt1[0] + u * (pt2[0] - pt1[0])), int(pt1[1] + u * (pt2[1] - pt1[1])))

    return np.sqrt((centroid[0] - closest_point[0]) ** 2 + (centroid[1] - closest_point[1]) ** 2)

# Function to check line crossings and log flags if conditions are met
def check_line_crossing(centroid, object_id, class_id, confidence, elapsed_time_seconds, frame, x1, y1, x2, y2):
    global crossed_objects_log

    # Initialize object log if it doesn't exist
    if object_id not in crossed_objects_log:
        crossed_objects_log[object_id] = {
            'class_id': class_id,
            'confidence': confidence,
            'crossings': 0,
            'last_crossing_time': 0,
            'crossing_times': [],
            'object_id': object_id,
            'flagged': False  # Track if the object has been flagged already
        }

    distance_threshold = 10  # Maximum allowed distance between centroid and line
    min_crossing_interval = 0.1  # Minimum time (in seconds) between crossings to avoid false counts

    # Check if the centroid is near any of the drawn lines (paths)
    for path in paths:
        if len(path['points']) > 1:
            for i in range(len(path['points']) - 1):
                pt1 = path['points'][i]
                pt2 = path['points'][i + 1]

                # Calculate the distance between the centroid and the line segment pt1-pt2
                distance = distance_from_line(pt1, pt2, centroid)

                if distance <= distance_threshold:  # If within threshold, log crossing
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')  # Get current time

                    # Prevent multiple rapid crossings by ensuring a minimum time interval between crossings
                    if elapsed_time_seconds - crossed_objects_log[object_id]['last_crossing_time'] >= min_crossing_interval:
                        crossed_objects_log[object_id]['crossings'] += 1
                        crossed_objects_log[object_id]['crossing_times'].append(elapsed_time_seconds)
                        crossed_objects_log[object_id]['last_crossing_time'] = elapsed_time_seconds  # Update last crossing time

                    # Log only if the object has crossed **any line** more than 4 times and hasn't been flagged
                    if crossed_objects_log[object_id]['crossings'] > 4 and not crossed_objects_log[object_id]['flagged']:
                        # Calculate the time difference between the first and last crossing
                        time_difference = crossed_objects_log[object_id]['crossing_times'][-1] - crossed_objects_log[object_id]['crossing_times'][0]

                        # Flag only if the 4 crossings happened within 4 seconds
                        if time_difference <= 4:  # If 4 crossings in less than 4 seconds, log it
                            with open('crossing_log.txt', 'a') as f:
                                f.write(f"{timestamp} - Violation type: 1 - ALERT: Object ID: {object_id} (Class: {class_id}) crossed more than 4 lines in {time_difference:.2f} seconds\n")

                            # Draw a red box around the object to indicate a violation
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                            # Mark the object as flagged so it doesn't get flagged again
                            crossed_objects_log[object_id]['flagged'] = True


# Function to process video and check line crossings
def process_video(video_file):
    crossed_objects_log.clear()  # Reset the log for each new video

    cap = cv2.VideoCapture(video_file)
    
    # Temporary file to save the processed video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_file = tfile.name

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer to save processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the predefined paths (lines) on each frame
        for path in paths:
            if len(path['points']) > 1:
                for i in range(len(path['points']) - 1):
                    pt1 = path['points'][i]
                    pt2 = path['points'][i + 1]
                    # Draw polyline in green color
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # YOLOv8 inference
        results = model(frame, conf=0.35, iou=0.5)
        detections = []

        # Extract detections from YOLOv8 results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf.item())
                class_id = int(box.cls)
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

        # Update DeepSORT tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = track.to_tlbr().astype(int)
            centroid_x, centroid_y = calculate_centroid([x1, y1, x2, y2])

            # Check for line crossings
            check_line_crossing((centroid_x, centroid_y), track_id, track.det_class, confidence, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame, x1, y1, x2, y2)

            # Draw bounding boxes around detected objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Write processed frame to output video
        out.write(frame)

    cap.release()
    out.release()
    
    return output_file

# Function to create a download button for files in Streamlit
def download_button(file_path, file_label, download_label):
    with open(file_path, "rb") as file:
        file_data = file.read()
    encoded_file = base64.b64encode(file_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{encoded_file}" download="{file_label}">{download_label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# Function to parse log file for violations
def parse_log_file(log_file):
    violations = []
    with open(log_file, 'r') as file:
        for line in file:
            if "plate" in line:
                print(f"Parsing line: {line.strip()}")
                match = re.search(r"plate\s+([\w\s]+).*?Class ID (\d+)", line)
                if match:
                    plate_number = match.group(1).strip()  # Capture the plate number
                    class_id = match.group(2).strip()  # Capture the Class ID
                    print(f"Matched Plate: {plate_number}, Class ID: {class_id}")
                    violations.append((plate_number, class_id))
    return violations

# Inserting violations into the database
def insert_violation_to_db(violations):
    conn = sqlite3.connect('lane_guard_DB.db')
    cursor = conn.cursor()

    # Insert data into detected_violation
    for violation in violations:
        plate, class_id = violation
        fine_amount = 3000.0  # As per the logic
        description = ""
        if class_id == '2':
            description = 'مخالفة مسار: تحريك المركبة في أكثر من مسار خلال مده زمنيه قصيره'
        else:
            description = 'مخالفة قيادة المركبات في مسار غير مخصصة لهم'

        cursor.execute('''
            INSERT INTO detected_violation (id_citizen, description, fine_amount)
            VALUES (?, ?, ?)
        ''', (plate, description, fine_amount))

    conn.commit()
    conn.close()

# Email functions for sending violation alerts
def generate_email_body(id_citizen, name, violation_description, plate_number, fine_amount):
    return f"""
تم قيد مخالفة:

{violation_description}

على سجل رقم: {id_citizen}

المالك: {name}

على السيارة: {plate_number}

قيمة المخالفة: {fine_amount} ريال

نرجو منكم مراجعة هذه المخالفة والالتزام بقوانين المرور حفاظاً على سلامتكم وسلامة الآخرين على الطريق.

مع خالص التحية،
- LaneGuard
"""

def send_email(email_address, id_citizen, name, violation_description, plate_number, fine_amount):
    SCOPES = ['https://www.googleapis.com/auth/gmail.send']
    creds = None

    # Check if token exists, otherwise authenticate user
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    # Build the Gmail API service
    service = build('gmail', 'v1', credentials=creds)

    # Generate the email body
    body = generate_email_body(id_citizen, name, violation_description, plate_number, fine_amount)

    # Correct subject line without encoding it again
    subject = f"تم قيد مخالفة: {name}"

    # Create a MIMEText object for the email body with proper UTF-8 encoding
    message = email.mime.text.MIMEText(body, _charset="UTF-8")
    message['to'] = email_address
    message['from'] = os.getenv('SENDER_EMAIL')  # Make sure the sender email is set in your environment
    message['subject'] = subject

    # Encode the email message into base64 format (from bytes, not string)
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    
    try:
        # Send the email via the Gmail API
        service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
        print(f"Email successfully sent to {name} ({email_address}) about {violation_description}")
    except Exception as e:
        print(f"Failed to send email to {email_address}. Error: {e}")

# Function to check the database for violations and send emails
def check_and_send_emails():
    conn = None
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('lane_guard_DB.db')
        cursor = conn.cursor()

        # Fetch all violations from the 'detected_violation' table
        cursor.execute('''SELECT id_citizen, description, fine_amount FROM detected_violation''')
        violations = cursor.fetchall()

        # Loop through all the violations and send emails
        for violation in violations:
            id_citizen, violation_description, fine_amount = violation

            # Fetch citizen details from the 'citizen' table
            cursor.execute('''SELECT name, email, plate_number FROM citizen WHERE id_citizen = ?''', (id_citizen,))
            citizen_info = cursor.fetchone()

            # If citizen details are found, send the email
            if citizen_info:
                name, email, plate_number = citizen_info
                send_email(email, id_citizen, name, violation_description, plate_number, fine_amount)

    except Exception as e:
        print(f"Error accessing the database: {e}")
    
    finally:
        if conn:
            conn.close()

# Streamlit app layout
st.title("YOLOv8 Object Detection and Line Crossing Tracker with Email Alerts")
st.write("Upload a video file to process.")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Store video temporarily for processing
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.write("Processing video...")
    processed_video_path = process_video(tfile.name)

    st.write("Here is the processed video:")
    
    # Display the processed video in Streamlit
    with open(processed_video_path, 'rb') as video_file:
        st.video(video_file.read())

    # Create a download button for the processed video
    st.write("Download the processed video:")
    download_button(processed_video_path, "processed_video.mp4", "Download Processed Video")

    # Check if the log file exists and allow it to be downloaded
    if os.path.exists('crossing_log.txt'):
        st.write("Download the crossing log:")
        download_button('crossing_log.txt', 'crossing_log.txt', 'Download Crossing Log')

    st.write("Crossing Log (showing only violations with >4 crossings):")
    st.json({obj_id: obj for obj_id, obj in crossed_objects_log.items() if obj['crossings'] > 4})

# Load environment variables (make sure SENDER_EMAIL is set in the environment)
load_dotenv()

# Process the log file and insert violations into DB
violations = parse_log_file('crossing_log.txt')
insert_violation_to_db(violations)

# Check the database and send emails
check_and_send_emails()