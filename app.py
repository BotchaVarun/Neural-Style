from flask import Flask, render_template, request, jsonify, send_file
import os
from PIL import Image, ImageFilter, ImageOps
import io
import base64
from werkzeug.utils import secure_filename
import uuid
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Define predefined style mappings
PREDEFINED_STYLES = {
    'monalisa': 'static/monolisa.webp',
    'picasso': 'static/picaso.jpg',
    'stary': 'static/stary.webp',
    'oilpainting1': 'static/oil-painting-1.webp',
    'oilpainting2': 'static/oil-painting-2.webp',
    'oilpainting3': 'static/oil-painting-3.webp',
    'anime': 'static/anmie1.jpg'
}

# Global variable to store the TensorFlow Hub model
hub_module = None

def load_style_transfer_model():
    """Load the TensorFlow Hub style transfer model"""
    global hub_module
    if hub_module is None:
        try:
            print("Loading TensorFlow Hub style transfer model...")
            hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            hub_module = None
    return hub_module

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_predefined_style_image(style_key):
    """Load a predefined style image from the static folder"""
    try:
        if style_key in PREDEFINED_STYLES:
            style_path = PREDEFINED_STYLES[style_key]
            if os.path.exists(style_path):
                return Image.open(style_path)
            else:
                print(f"Style image not found: {style_path}")
                return None
        return None
    except Exception as e:
        print(f"Error loading predefined style image: {e}")
        return None

def apply_filter(image, filter_type, filter_image=None):
    """Apply the selected filter to the image."""
    if filter_type == 'grayscale':
        return ImageOps.grayscale(image)
    elif filter_type == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=2))
    elif filter_type == 'edge-detect':
        return image.filter(ImageFilter.FIND_EDGES)
    elif filter_type == 'sharpen':
        return image.filter(ImageFilter.SHARPEN)
    elif filter_type == 'emboss':
        return image.filter(ImageFilter.EMBOSS)
    elif filter_type in PREDEFINED_STYLES:
        # Load predefined style image
        style_image = load_predefined_style_image(filter_type)
        if style_image:
            return apply_neural_style_transfer(image, style_image)
        else:
            print(f"Could not load predefined style: {filter_type}")
            return image
    elif filter_type == 'other' and filter_image:
        return apply_neural_style_transfer(image, filter_image)
    else:
        return image

def preprocess_image_for_tf(image, max_size=512):
    """Preprocess PIL image for TensorFlow"""
    # Resize image while maintaining aspect ratio
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    
    # Ensure image has 3 channels (RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Normalize to [0, 1] and add batch dimension
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def apply_neural_style_transfer(content_image, style_image):
    """Apply neural style transfer using TensorFlow Hub model"""
    try:
        # Load the model if not already loaded
        model = load_style_transfer_model()
        if model is None:
            print("Model not available, falling back to simple blend")
            return apply_simple_blend(content_image, style_image)
        
        print("Applying neural style transfer...")
        
        # Convert PIL images to RGB if necessary
        if content_image.mode != 'RGB':
            content_image = content_image.convert('RGB')
        if style_image.mode != 'RGB':
            style_image = style_image.convert('RGB')
        
        # Preprocess images
        content_array = preprocess_image_for_tf(content_image.copy())
        style_array = preprocess_image_for_tf(style_image.copy(), max_size=256)
        
        # Apply style transfer
        print("Running style transfer model...")
        outputs = model(tf.constant(content_array), tf.constant(style_array))
        stylized_image = outputs[0]
        
        # Convert back to PIL Image
        result_array = np.squeeze(stylized_image.numpy())
        result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        result_image = Image.fromarray(result_array)
        
        # Resize to match original content image size
        result_image = result_image.resize(content_image.size, Image.Resampling.LANCZOS)
        
        print("Style transfer completed successfully!")
        return result_image
        
    except Exception as e:
        print(f"Error in neural style transfer: {e}")
        # Fallback to simple blending if neural style transfer fails
        return apply_simple_blend(content_image, style_image)

def apply_simple_blend(base_image, filter_image):
    """Fallback: Apply custom filter using simple image overlay blending."""
    try:
        # Resize filter image to match base image size
        filter_img = filter_image.resize(base_image.size, Image.Resampling.LANCZOS)
        
        # Convert both images to RGBA for blending
        if base_image.mode != 'RGBA':
            base_image = base_image.convert('RGBA')
        if filter_img.mode != 'RGBA':
            filter_img = filter_img.convert('RGBA')
        
        # Create a blended image using multiply blend mode
        result = Image.new('RGBA', base_image.size)
        
        # Blend the images with 30% opacity for better results
        result = Image.blend(base_image, filter_img, 0.3)
        
        # Convert back to RGB
        if result.mode == 'RGBA':
            background = Image.new('RGB', result.size, (255, 255, 255))
            background.paste(result, mask=result.split()[-1])
            result = background
        
        return result
    except Exception as e:
        print(f"Error in simple blend: {e}")
        return base_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        filter_type = request.form.get('filter')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        if not filter_type:
            return jsonify({'error': 'No filter selected'}), 400
        
        # Handle filter image for 'other' option
        filter_image = None
        if filter_type == 'other':
            if 'filterImage' not in request.files:
                return jsonify({'error': 'Filter image required for custom filter'}), 400
            
            filter_file = request.files['filterImage']
            if filter_file.filename == '':
                return jsonify({'error': 'No filter image selected'}), 400
            
            if not allowed_file(filter_file.filename):
                return jsonify({'error': 'Invalid filter image type'}), 400
            
            filter_image = Image.open(filter_file.stream)
        
        # Generate unique filename
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        
        # Open and process the image
        image = Image.open(file.stream)
        
        # Convert to RGB if necessary (for JPEG output)
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        print(f"Processing image with filter: {filter_type}")
        
        # Apply the selected filter
        processed_image = apply_filter(image, filter_type, filter_image)
        
        # Save processed image to memory
        img_io = io.BytesIO()
        processed_image.save(img_io, 'JPEG', quality=95)
        img_io.seek(0)
        
        # Convert to base64 for JSON response
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'filename': filename
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_image(filename):
    try:
        # For security, we'll regenerate the image instead of storing it
        # In a real app, you might want to temporarily store processed images
        return jsonify({'error': 'Download not implemented in this demo'}), 501
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Pre-load the model when starting the application
    print("Starting Flask application...")
    print("Note: The first style transfer may take longer as the model needs to be downloaded and loaded.")
    
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
