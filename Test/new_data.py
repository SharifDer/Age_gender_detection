import os
import cv2
import numpy as np
import tensorflow as tf

class FacePredictor:
    def __init__(self, model, data_handler):
        self.model = model
        self.data_handler = data_handler
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    def _process_face(self, face_roi, image_path):
        """Process a single face ROI and return prediction results"""
        try:
            target_size = (200, 200)  # Set based on your model input

            # Convert OpenCV BGR to RGB
            rgb_image = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Convert to Tensor
            image_tensor = tf.convert_to_tensor(rgb_image, dtype=tf.float32)

            # Resize
            image_tensor = tf.image.resize(image_tensor, target_size)

            # Normalize to [0, 1]
            image_tensor = tf.clip_by_value(image_tensor / 255.0, 0.0, 1.0)

            # Add batch dimension
            prepared_image = tf.expand_dims(image_tensor, axis=0)

            # Predict
            age_pred, gender_pred = self.model.predict(prepared_image, verbose=0)

            # Parse ground truth from filename (if available)
            filename = os.path.basename(image_path)
            parts = filename.split('_')
            has_truth = len(parts) >= 3

            return {
                'predicted': {
                    'age': float(age_pred[0][0]),
                    'gender': 'Female' if gender_pred[0] > 0.5 else 'Male'
                },
                'true': {
                    'age': int(parts[0]),
                    'gender': 'Male' if int(parts[1]) == 0 else 'Female'
                } if has_truth else None
            }
        except Exception as e:
            print(f"Face processing failed: {str(e)}")
            return None

    def predict_image(
        self,
        image_path,
        min_n,
        style_settings,
        output_path=None
    ):
        """
        Process all faces in image and return annotated image + predictions.

        Parameters:
        - image_path (str): Path to the input image.
        - min_n (int): Minimum neighbors for face detection.
        - style_settings (dict): Contains all styling options:
            {
                'font': cv2 font constant,
                'font_scale': float,
                'font_thickness': int,
                'text_color': (B, G, R),
                'bg_color': (B, G, R),
                'box_color_male': (B, G, R),
                'box_color_female': (B, G, R),
                'rectangle_thickness': int,
                'text_box_margin': int  # New parameter for spacing
            }
        - output_path (str, optional): Path to save the output image.
        """

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None, []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_n, minSize=(30, 30))

        if len(faces) == 0:
            print("No faces detected in the image")
            return img, []
        print("len face" , len(faces)) 
        predictions = []
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            result = self._process_face(face_roi, image_path)
            if not result:
                continue

            predictions.append(result)
            gender = result['predicted']['gender']
            age = result['predicted']['age']

            # Pick rectangle color based on gender
            box_color = style_settings['box_color_male'] if gender == 'Male' else style_settings['box_color_female']

            label = f"{gender}, Age: {age:.1f} Years"
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                style_settings['font'],
                style_settings['font_scale'],
                style_settings['font_thickness']
            )

            # Get margin from style settings (default to 0 if not specified)
            text_box_margin = style_settings.get('text_box_margin', 0)

            # Coordinates for label background with margin
            text_bg_x1 = x
            text_bg_y1 = y - text_height - baseline - 6 - text_box_margin
            text_bg_x2 = x + text_width + 10
            text_bg_y2 = y - text_box_margin  # This creates the space between box and text
            text_bg_y1 = max(0, text_bg_y1)  # Ensure it doesn't go above the image

            # Draw rectangle and label
            cv2.rectangle(img, (x, y), (x + w, y + h), box_color, style_settings['rectangle_thickness']
                          ,   )
            cv2.rectangle(img, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2),
                           style_settings['bg_color'],
                           -1)
            cv2.putText(
                img,
                label,
                (x +5, text_bg_y2 - 5),  # Text position within its background x width text_bg_ye height
                style_settings['font'],
                style_settings['font_scale'],
                style_settings['text_color'],
                style_settings['font_thickness']
            )

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img)

        return img, predictions

    # Original methods maintained for backward compatibility
    def predict_single(self, image_path, output_path='Data/extracted_faces/'):
        """Legacy method for single face prediction (maintained for compatibility)"""
        img, predictions = self.predict_image(image_path)
        return predictions[0] if predictions else None

    def print_prediction(self, prediction):
        """Print formatted prediction results"""
        if not prediction:
            print("No prediction available")
            return
            
        print("\nPrediction Results:")
        print(f"Age: {prediction['predicted']['age']:.1f} years")
        print(f"Gender: {prediction['predicted']['gender']}")
        
        if prediction['true']:
            print("\nTrue Values:")
            print(f"Age: {prediction['true']['age']}")
            print(f"Gender: {prediction['true']['gender']}")

    def sample_predictions(self, test_data, num_samples=3):
        """Get sample predictions from test dataset"""
        samples = []
        for batch in test_data.take(1):
            images, labels = batch
            age_pred, gender_pred = self.model.predict(images, verbose=0)[:2]
            
            for i in range(min(num_samples, len(images))):
                samples.append({
                    'true_age': labels['age'][i].numpy(),
                    'true_gender': 'Male' if labels['gender'][i].numpy() == 0 else 'Female',
                    'pred_age': float(age_pred[i][0]),
                    'pred_gender': 'Female' if gender_pred[i] > 0.5 else 'Male'
                })
        return samples