import tensorflow as tf
class Trainer:
    def __init__(self, model, callbacks=None):
            self.model = model
            self.callbacks = callbacks or [
                tf.keras.callbacks.EarlyStopping(patience=5),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    verbose=1
                )
            ]

    def train(self, train_data, val_data, epochs=10):
        """Train model with proper callbacks"""
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=self.callbacks
        )
        return history

    def save_model(self, path):
        self.model.save(path)