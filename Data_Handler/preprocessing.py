import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class DataHandler:
    def __init__(self, img_size=(200, 200), val_size=0.15, test_size=0.15):
        """
        Args:
            img_size: Target image dimensions
            val_size: Fraction for validation set (0-1)
            test_size: Fraction for test set (0-1)
        """
        self.img_size = img_size
        self.val_size = val_size
        self.test_size = test_size
        self.label_map = {
            'gender': {0: 'Male', 1: 'Female'},
            # 'race': {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Other'}
        }

    def _parse_filename(self, filename):
        """Extract metadata from filename format: AGE_GENDER_RACE_*.jpg"""
        parts = tf.strings.split(tf.strings.split(filename, os.path.sep)[-1], '_')
        return {
            'age': tf.strings.to_number(parts[0], tf.float32),  # Changed to float for regression
            'gender': tf.strings.to_number(parts[1], tf.int32),
            # 'race': tf.strings.to_number(parts[2], tf.int32)
        }

    def _preprocess_image(self, image, augment=False):
        """Enhanced preprocessing with optional augmentation"""
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, self.img_size)
        # image = tf.image.rgb_to_grayscale(image)
        image = tf.clip_by_value(image / 255.0, 0, 1)
        
        if augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.1)
        return image

    def create_datasets(self, image_dir, batch_size=32, augment_train=False):
        """
        Creates train/val/test datasets with proper stratification
        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        all_files = np.array(self._get_image_files(image_dir))
        
        # Extract labels for stratification
        labels = np.array([self._get_sample_label(f) for f in all_files])
        
        # First split: train + temp (val+test)
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            all_files, labels, 
            test_size=(self.val_size + self.test_size),
            stratify=labels
        )
        
        # Second split: val + test
        val_files, test_files = train_test_split(
            temp_files,
            test_size=self.test_size/(self.val_size + self.test_size),
            stratify=temp_labels
        )
        
        return (
            self._build_dataset(train_files, batch_size, shuffle=True, augment=augment_train),
            self._build_dataset(val_files, batch_size),
            self._build_dataset(test_files, batch_size)
        )

    def _get_sample_label(self, filename):
        """Get simplified label for stratification"""
        parts = os.path.basename(filename).split('_')
        return f"{parts[1]}_{parts[2]}"  # gender_race combination

    def _get_image_files(self, image_dir):
        """Get filtered list of image files with safety checks"""
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory {image_dir} does not exist")
            
        return [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def _build_dataset(self, files, batch_size, shuffle=False, augment=False):
        """Optimized dataset pipeline with optional augmentation"""
        def parse_item(filename):
            labels = self._parse_filename(filename)
            image = tf.io.read_file(filename)
            return self._preprocess_image(image, augment), labels
        
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(parse_item, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            ds = ds.shuffle(buffer_size=2*batch_size, reshuffle_each_iteration=True)
            
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def prepare_single_image(self, image_path):
        """Prepare single image for prediction with error handling"""
        try:
            image = tf.io.read_file(image_path)
            return tf.expand_dims(self._preprocess_image(image), axis=0)
        except:
            raise ValueError(f"Could not process image: {image_path}")