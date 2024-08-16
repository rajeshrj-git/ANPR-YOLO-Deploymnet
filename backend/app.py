from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from pymongo import MongoClient
import os
from model import process_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'annotated_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['anpr']
collection = db['results']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return redirect(request.url)
            if file:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                
                try:
                    file.save(filepath)

                    # Process the image
                    output_image_path, extracted_text = process_image(filepath, app.config['ANNOTATED_FOLDER'])

                    # Save the result to MongoDB
                    collection.insert_one({
                        'input_image': filepath,
                        'annotated_image': output_image_path,
                        'extracted_text': extracted_text
                    })

                    return render_template('index.html', image_url=url_for('serve_annotated_image', filename=os.path.basename(output_image_path)), extracted_text=extracted_text)

                except Exception as e:
                    return f"An error occurred: {e}", 500
    
    return render_template('index.html')

@app.route('/annotated_images/<filename>')
def serve_annotated_image(filename):
    return send_from_directory(app.config['ANNOTATED_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(ANNOTATED_FOLDER):
        os.makedirs(ANNOTATED_FOLDER)
    app.run(debug=True)
