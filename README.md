# 🌿 House Plant Species Identifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1.1-black?style=for-the-badge&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c?style=for-the-badge&logo=pytorch)

A complete end-to-end web application that leverages a deep learning model to identify the species of house plants from user-uploaded images. This project demonstrates skills in machine learning model integration, backend development with Flask, and frontend presentation.

##  Demo

![House Plant Classifier Demo](./assets/app_sc1.png)


---

## Table of Contents

- Key Features
- Technology Stack
- System Architecture
- Model Details
- Results
- Local Setup & Installation
- Future Enhancements

---

## Key Features

*   **Image-Based Classification**: Users can upload an image (`.jpg`, `.png`, etc.) of a house plant.
*   **Instant Predictions**: The application returns the predicted species name along with a confidence score.
*   **Interactive UI**: A clean, responsive, and user-friendly interface built with Flask and HTML/CSS.
*   **Efficient Backend**: The Flask server handles file uploads and model inference efficiently.
*   **Pre-trained Deep Learning Model**: Utilizes a powerful `EfficientNet-B0` model for accurate and fast predictions.

---

## Technology Stack

This project integrates a variety of modern technologies for web development and machine learning:

*   **Backend**: **Python**, **Flask**
*   **Machine Learning**: **PyTorch**, **Torchvision**
*   **Frontend**: **HTML5**, **CSS3**
*   **Core Python Libraries**: **Pillow** (for image manipulation), **NumPy**

---

## System Architecture

The application follows a simple yet robust client-server architecture:

1.  **Frontend (Client-Side)**: A user accesses the web interface, built with HTML and styled with CSS, and uploads an image via a form.
2.  **Backend (Server-Side - Flask)**:
    *   A Flask route (`/`) receives the `POST` request containing the image file.
    *   The image is read into memory as bytes.
    *   The image bytes are passed to the prediction module.
3.  **Machine Learning Model (PyTorch)**:
    *   The `model.py` utility loads a pre-trained `EfficientNet-B0` model into memory once at application startup to ensure low latency for subsequent predictions.
    *   The uploaded image is preprocessed (resized, normalized, and converted to a tensor) to match the model's input requirements.
    *   The model performs inference on the preprocessed image.
    *   The highest probability output is identified and mapped to its corresponding class name.
4.  **Response**: The predicted species and confidence score are sent back to the Flask backend, which then renders the `index.html` template to display the results to the user.

---

## Model Details

*   **Model**: `EfficientNet-B0`
*   **Framework**: PyTorch
*   **Pre-training**: The model was fine-tuned on a diverse dataset of house plant images. The weights used in this application (`house_plant_classifier_v1.pth`) are the result of this training process.
*   **Classes**: The model is capable of identifying **47 different species** of common house plants. See `class_names.txt` for the full list.

---

## Results 📈

Our machine learning model was trained to identify various house plant species. The complete training and evaluation process is documented in the [Kaggle Notebook](https://github.com/MdEhsanulHaqueKanan/house-plant-species-identifier-machine-learning-flask-app/blob/main/notebook/house-plant-identification.ipynb).

### Summary of Model Performance

After training for 10 epochs, the model achieved the following key performance metrics on the test dataset:

* **Overall Accuracy:** **82%**
* **Final Validation Loss (from training logs):** 0.6232
* **Final Training Loss (from training logs):** 0.4358

### Detailed Classification Report (Test Set)

The comprehensive classification report on the test set provides a detailed breakdown of the model's precision, recall, and F1-score for each plant species, as well as overall averages:

| Metric       | Precision | Recall | F1-Score | Support |
| :----------- | :-------- | :----- | :------- | :------ |
| **Macro Avg** | **0.82** | **0.82** | **0.81** | **1478**|
| **Weighted Avg** | **0.83** | **0.82** | **0.82** | **1478**|

---

## Local Setup & Installation

To run this application on your local machine, please follow these steps:

**Prerequisites:**
*   Python 3.9 or higher
*   `pip` package installer

**1. Clone the Repository**
```bash
git clone https://github.com/MdEhsanulHaqueKanan/house-plant-species-identifier-machine-learning-flask-app.git
cd house-plant-species-identifier-machine-learning-flask-app
```

**2. Create and Activate a Virtual Environment**

*On Windows:*
```bash
python -m venv venv
.\venv\Scripts\activate
```

*On macOS/Linux:*
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the Model File**
The pre-trained model weights (`house_plant_classifier_v1.pth`) are not tracked by Git. You must download it manually and place it in the `models/` directory for the application to work correctly.

**[Download the model file from Google Drive here](https://drive.google.com/file/d/1qAICG_FhkrZAnGURVBRlp3KFsAgQ092t/view?usp=sharing)**

**5. Run the Application**
```bash
flask run
```

The application will be available at `http://127.0.0.1:5000` in your web browser.

---

## Future Enhancements

*   **Plant Care Information**: Upon successful identification, provide users with basic care tips for the predicted plant species (e.g., watering frequency, light requirements).
*   **REST API**: Expose the prediction logic as a RESTful API endpoint for other applications to consume.
*   **Cloud Deployment**: Deploy the application to a cloud platform like Heroku, AWS Elastic Beanstalk, or Google Cloud Run for public access.
*   **CI/CD Pipeline**: Implement a continuous integration and deployment pipeline using GitHub Actions to automate testing and deployment.

---

Developed by Md. Ehsanul Haque Kanan