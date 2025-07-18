# app.py

from flask import Flask, render_template, request, redirect, url_for
import base64
import model as model_predictor # Import our model utility file

app = Flask(__name__)

# --- MODEL LOADING ---
# Load the model, class names, and transforms only once when the app starts
# This is crucial for performance
print("Loading model and classes...")
model, class_names, auto_transforms = model_predictor.load_model_and_classes()
print("Model and classes loaded successfully.")


# --- ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            # Read the image file as bytes
            image_bytes = file.read()
            
            # Make prediction
            predicted_class, confidence = model_predictor.predict(
                image_bytes=image_bytes,
                model=model,
                class_names=class_names,
                transform=auto_transforms
            )
            
            # Encode image bytes to base64 string for display
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            
            # Render the template with the prediction result
            return render_template(
                'index.html',
                prediction=predicted_class,
                confidence=f"{(confidence * 100):.2f}%", # Format as percentage
                image_data=image_data)
            
    # For a GET request, just render the initial page
    return render_template('index.html', prediction=None, confidence=None, image_data=None)

if __name__ == '__main__':
    app.run(debug=True)