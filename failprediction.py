import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os

class NodeFailurePredictor:
    """
    LSTM Model for predicting node failure probability from RSSI and link quality signals
    """
    
    def __init__(self, model_path='models/failure_model.h5', window_size=10):
        self.model_path = model_path
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.trained = False
        
    def generate_synthetic_training_data(self, samples=1000, sequence_length=10):
        """
        Generate synthetic RSSI data with labels (0=stable, 1=failure)
        Signal drop pattern: Normal(-50 to -70) -> Failing(-70 to -95)
        """
        X_data = []
        y_data = []
        
        for _ in range(samples):
            # Stable signal pattern (80% of data)
            if np.random.rand() > 0.2:
                signal = np.random.uniform(-70, -50, sequence_length)
                label = 0  # Stable
            # Failing signal pattern (20% of data)
            else:
                signal = np.linspace(-70, -95, sequence_length) + np.random.normal(0, 2, sequence_length)
                label = 1  # Failure
            
            X_data.append(signal)
            y_data.append(label)
        
        X_data = np.array(X_data).reshape(samples, sequence_length, 1)
        y_data = np.array(y_data)
        
        # Normalize
        X_data = (X_data + 100) / 100  # Normalize -100 to 0 -> 0 to 1
        
        return X_data, y_data
    
    def build_model(self):
        """Build and compile LSTM model"""
        self.model = Sequential([
            LSTM(64, input_shape=(self.window_size, 1), activation='relu', return_sequences=True),
            LSTM(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Probability of failure (0-1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def train(self, epochs=50, validation_split=0.2):
        """Train the LSTM model on synthetic data"""
        print("[INFO] Generating synthetic training data...")
        X_train, y_train = self.generate_synthetic_training_data(samples=1000, sequence_length=self.window_size)
        
        print("[INFO] Building LSTM model...")
        self.build_model()
        
        print("[INFO] Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            verbose=1
        )
        
        self.trained = True
        print("[SUCCESS] Model training complete!")
        return history
    
    def save_model(self):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if self.model:
            self.model.save(self.model_path)
            print(f"[SUCCESS] Model saved to {self.model_path}")
    
    def load_model(self):
        """Load pre-trained model from disk"""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            self.trained = True
            print(f"[SUCCESS] Model loaded from {self.model_path}")
            return True
        else:
            print(f"[WARNING] Model not found at {self.model_path}")
            return False
    
    def predict_failure_probability(self, rssi_stream):
        """
        Predict failure probability from RSSI stream
        Input: array of normalized RSSI values (shape: (window_size,))
        Output: probability (0-1)
        """
        if not self.trained or self.model is None:
            raise RuntimeError("Model not trained or loaded")
        
        # Prepare input: reshape to (1, window_size, 1)
        input_data = np.array(rssi_stream).reshape(1, self.window_size, 1)
        
        # Predict
        prediction = self.model.predict(input_data, verbose=0)
        return float(prediction[0][0])
    
    def check_link_health(self, rssi_stream, threshold=0.70):
        """
        Check link health and return action
        Returns: "LINK_STABLE" or "TRIGGER_PROACTIVE_REROUTE"
        """
        if not self.trained:
            return "LINK_UNKNOWN"
        
        prob = self.predict_failure_probability(rssi_stream)
        
        if prob > threshold:
            return "TRIGGER_PROACTIVE_REROUTE"
        return "LINK_STABLE"


# Example usage
if __name__ == "__main__":
    predictor = NodeFailurePredictor()
    
    # Train new model
    history = predictor.train(epochs=50)
    
    # Save for later use
    predictor.save_model()
    
    # Test prediction
    test_rssi = [(-60 + 100) / 100, (-62 + 100) / 100, (-65 + 100) / 100,
                 (-68 + 100) / 100, (-70 + 100) / 100, (-75 + 100) / 100,
                 (-80 + 100) / 100, (-85 + 100) / 100, (-90 + 100) / 100, (-95 + 100) / 100]
    
    prob = predictor.predict_failure_probability(test_rssi)
    action = predictor.check_link_health(test_rssi)
    
    print(f"\n[TEST] Failure Probability: {prob:.2%}")
    print(f"[TEST] Action: {action}")