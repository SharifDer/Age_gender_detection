import tensorflow as tf
import os
import cv2
import numpy as np

class Tester:
    def __init__(self, model, data_handler):
        self.model = model
        self.data_handler = data_handler
        
# In Tester.evaluate()
    def evaluate(self, test_data):
        results = self.model.evaluate(test_data, verbose=0)
        return {
            'age_mae': results[3],  # 4th item in model.metrics_names
            'gender_accuracy': results[4]  # 5th item
        }


    def print_results(self, metrics):
        """Print formatted results (age and gender only)"""
        print("\nTest Results:")
        if metrics['age_mae'] is not None:
            print(f"Age MAE: {metrics['age_mae']:.2f} years")
        else:
            print("Age MAE metric not available")
            
        if metrics['gender_accuracy'] is not None:
            print(f"Gender Accuracy: {metrics['gender_accuracy']:.2%}")

    def predict_single(self, image_path, output_path='extracted_faces/'):
        """Make prediction on a single image file (age and gender only)"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print("No faces detected")
            return None
            
        # Extract first face only
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        
        # Save the extracted face
        os.makedirs(output_path, exist_ok=True)
        face_filename = f"{output_path}face_0.jpg"
        cv2.imwrite(face_filename, face)
        
        try:
            # Preprocess the face image
            image = self.data_handler.prepare_single_image(face_filename)
            
            # Get predictions (age and gender only)
            predictions = self.model.predict(image, verbose=0)
            
            # Handle different model output formats:
            if len(predictions) == 2:  # If model outputs only age and gender
                age_pred, gender_pred = predictions
            else:  # If model outputs age, gender, race (but we ignore race)
                age_pred, gender_pred, _ = predictions
            
            # Convert to human-readable format
            gender = 'Female' if gender_pred[0] > 0.5 else 'Male'
            
            # Try to extract true values from filename (age and gender only)
            filename = os.path.basename(image_path)
            parts = filename.split('_')
            true_age = int(parts[0]) if len(parts) >= 3 else None
            true_gender = int(parts[1]) if len(parts) >= 3 else None
            
            return {
                'predicted': {
                    'age': float(age_pred[0][0]),
                    'gender': gender
                },
                'true': {
                    'age': true_age,
                    'gender': 'Female' if true_gender == 1 else 'Male'
                } if true_age is not None else None
            }
            
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return None

    def print_prediction(self, prediction):
        """Print formatted prediction results (age and gender only)"""
        if prediction is None:
            print("No prediction available")
            return
            
        print("\nPrediction Results:")
        print(f"Age: {prediction['predicted']['age']:.1f} years")
        print(f"Gender: {prediction['predicted']['gender']}")
        
        if prediction['true'] is not None:
            print("\nTrue Values:")
            print(f"Age: {prediction['true']['age']}")
            print(f"Gender: {prediction['true']['gender']}")

    def sample_predictions(self, test_data, num_samples=3):
        """Get sample predictions from test dataset"""
        samples = []
        for batch in test_data.take(1):  # Take one batch
            images, labels = batch
            preds = self.model.predict(images, verbose=0)
            
            if len(preds) == 2:
                age_preds, gender_preds = preds
            else:
                age_preds, gender_preds, _ = preds
                
            for i in range(min(num_samples, len(images))):
                samples.append({
                    'true_age': labels['age'][i].numpy(),
                    'true_gender': 'Female' if labels['gender'][i].numpy() == 1 else 'Male',
                    'pred_age': float(age_preds[i][0]),
                    'pred_gender': 'Female' if gender_preds[i] > 0.5 else 'Male'
                })
        return samples
