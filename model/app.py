from flask import Flask, request, jsonify
from model import process_image
import os

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    input_image_path = os.path.join('/tmp', file.filename)
    file.save(input_image_path)

    try:
        output_folder = '/tmp'  # Output folder for the annotated image
        output_image_path, extracted_text = process_image(input_image_path, output_folder)

        return jsonify({
            'output_image_path': output_image_path,
            'extracted_text': extracted_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
