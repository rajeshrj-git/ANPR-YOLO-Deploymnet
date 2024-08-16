import os
from datetime import datetime
import cv2
import pytesseract
from ultralytics import YOLO

# Configure the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as needed

# Load pretrained YOLOv8 model
model = YOLO('best.pt')

def process_image(input_image_path, output_folder):
    """
    Processes the image using YOLO and Tesseract, saves the annotated image,
    and returns the path to the saved image and extracted text.
    """
    # Run YOLO inference
    results = model.predict(input_image_path, save=False, imgsz=320, conf=0.2)

    # Load the image with OpenCV
    image = cv2.imread(input_image_path)

    if image is None:
        raise ValueError("Image not loaded correctly. Check the path.")

    extracted_text = ''
    # Process each detected bounding box
    for result in results:
        detections = result.cpu().numpy() if hasattr(result, 'cpu') else result.numpy()

        for detection in detections:
            if len(detection) >= 6:
                x_min, y_min, x_max, y_max, conf, cls = detection[:6]

                # Convert to integers
                x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))

                # Crop the detected number plate region from the image
                plate_img = image[y_min:y_max, x_min:x_max]

                # Convert the cropped image to grayscale (improves OCR accuracy)
                gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

                # Use pytesseract to extract text
                extracted_text = pytesseract.image_to_string(gray_plate_img, config='--psm 8').strip().upper()

                # Annotate the detected text on the image
                cv2.putText(image, extracted_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw the bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # Extract the base filename (without extension) from the input path
    base_filename = os.path.splitext(os.path.basename(input_image_path))[0]

    # Generate a unique filename using the base filename and current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{base_filename}_annotated_{timestamp}.png"

    # Define the output path
    output_image_path = os.path.join(output_folder, output_filename)

    # Save the output image with annotations
    cv2.imwrite(output_image_path, image)

    return output_image_path, extracted_text
