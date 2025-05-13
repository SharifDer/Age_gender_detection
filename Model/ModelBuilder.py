from keras import layers, Model
import tensorflow as tf
class ModelBuilder:
    @staticmethod
    def build_model(input_shape=(200, 200, 3)):
        """Build multi-task model with EfficientNet backbone"""
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # Backbone
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        backbone.trainable = True
        x = backbone(inputs , training=False)
        features = layers.GlobalAvgPool2D()(x)
        
        # Task-specific heads
        age = layers.Dense(128, activation='relu')(features)
        age = layers.Dense(1, name='age')(age)
        
        gender = layers.Dense(64, activation='relu')(features)
        gender = layers.Dense(1, activation='sigmoid', name='gender')(gender)
        
        # race = layers.Dense(64, activation='relu')(features)
        # race = layers.Dense(5, activation='softmax', name='race')(race)
        
        return Model(inputs=inputs, outputs=[age, gender])

    @staticmethod
# In ModelBuilder.py
    def compile_model(model, learning_rate=0.00001):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss={
                'age': 'mse',
                'gender': 'binary_crossentropy'
            },
            metrics={
                'age': [tf.keras.metrics.MeanAbsoluteError(name='age_mae')],
                'gender': [tf.keras.metrics.BinaryAccuracy(name='gender_accuracy')]
            }
        )
        return model