from ultralytics import YOLO
import cv2
import pytesseract
import tempfile
import os

# Configure the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as needed

# Load pretrained YOLOv8 model
model = YOLO('best.pt')

# Path to the input video file
input_video_path = r"C:\Users\rajes\Downloads\Details behind Temporary License Plate in New Cars. How many of you know this_ #motowagon #carfacts.mp4"
output_video_path = r"C:\Users\rajes\Datascience_jp\Car_Number_Plate_200MB\train\car_images\output_video.mp4"  # Changed to .mp4

# Open the video file
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise ValueError("Error opening video file. Check the path.")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed codec to 'mp4v' for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no more frames

    # Save the frame as a temporary image file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image_file:
        temp_image_path = temp_image_file.name
        cv2.imwrite(temp_image_path, frame)

    # Run YOLO inference on the temporary image file
    results = model.predict(temp_image_path, save=False, imgsz=320, conf=0.2)

    # Print the results for debugging
    print("Results:", results)

    # Process each detected bounding box
    for result in results:
        # Convert tensor to numpy array for easier handling
        detections = result.cpu().numpy() if hasattr(result, 'cpu') else result.numpy()
        
        # Print detections to understand structure
        print("Detections:", detections)

        # Iterate over detections
        for detection in detections:
            if len(detection) >= 6:  # Check if detection has required number of elements
                x_min, y_min, x_max, y_max, conf, cls = detection[:6]

                # Convert to integers
                x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))

                # Crop the detected number plate region from the frame
                plate_img = frame[y_min:y_max, x_min:x_max]

                # Convert the cropped image to grayscale (improves OCR accuracy)
                gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

                # Use pytesseract to extract text
                extracted_text = pytesseract.image_to_string(gray_plate_img, config='--psm 8').upper()  # --psm 8 is for sparse text

                # Annotate the detected text on the frame
                cv2.putText(frame, extracted_text.strip(), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw the bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # Write the annotated frame to the output video
    out.write(frame)

    # Clean up the temporary image file
    os.remove(temp_image_path)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
