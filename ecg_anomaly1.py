import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras import layers, losses, Model
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

# Suppress warnings
warnings.filterwarnings('ignore')

# Set Streamlit page configuration
st.set_page_config(page_title="ECG Anomaly Detection", page_icon="💓", layout="wide")

# Load data function
@st.cache_data
def load_data(file=None):
    try:
        if file is not None:
            df = pd.read_csv(file)
        else:
            df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Define and load model function
@st.cache_resource
def load_model():
    class Detector(Model):
        def __init__(self):
            super(Detector, self).__init__()
            self.encoder = tf.keras.Sequential([
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(8, activation='relu')
            ])
            self.decoder = tf.keras.Sequential([
                layers.Dense(16, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(140, activation='sigmoid')
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    model = Detector()
    model.compile(optimizer='adam', loss='mae')
    return model

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your ECG data (CSV)", type=["csv"])

# Load data
df = load_data(uploaded_file)
if df is not None:
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)
    min_val, max_val = np.min(train_data), np.max(train_data)
    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)
    
    # Ensure correct data type (float32)
    train_data, test_data = map(lambda x: tf.cast(x, dtype=tf.float32), [train_data, test_data])
    train_labels, test_labels = train_labels.astype(bool), test_labels.astype(bool)

    # Separate normal and anomalous data
    n_train_data, n_test_data = train_data[train_labels], test_data[test_labels]
    an_train_data, an_test_data = train_data[~train_labels], test_data[~test_labels]

    # Load and train the autoencoder model
    autoencoder = load_model()

    # Train model with explicit casting
    autoencoder.fit(n_train_data, n_train_data, epochs=20, batch_size=64, validation_data=(n_test_data, n_test_data))

else:
    st.warning("No ECG data available. Please upload a dataset.")

# Define LIME Explainer
if df is not None:
    explainer = LimeTabularExplainer(
        training_data=n_train_data.numpy(),
        mode="regression",
        feature_names=[f"Feature {i}" for i in range(n_train_data.shape[1])],
        discretize_continuous=True
    )

# Function to plot residual plot with LIME importance overlay
def plot_residual_with_lime(data, index):
    fig, ax = plt.subplots()
    enc_img = autoencoder.encoder(data)
    dec_img = autoencoder.decoder(enc_img)
    residuals = data[index] - dec_img[index]
    ax.scatter(range(len(residuals)), residuals, c='red', label='Residuals')
    ax.axhline(0, color='black', linestyle='--')
    
    # Generate LIME explanation
    data_instance = data[index].numpy().reshape(1, -1)
    def model_predict(input_data):
        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
        return autoencoder.predict(input_data).reshape(-1, 140)
    
    exp = explainer.explain_instance(data_instance.flatten(), model_predict)
    lime_weights = dict(exp.as_list())
    
    # Overlay LIME importance on residual plot
    for i, weight in lime_weights.items():
        ax.scatter(int(i.split()[-1]), residuals[int(i.split()[-1])], color='blue', s=abs(weight) * 100, alpha=0.6, label='LIME Importance' if i == 0 else "")
    
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Residual (Input - Reconstruction)")
    ax.legend()
    st.pyplot(fig)

# Sidebar inputs
st.sidebar.title("ECG Anomaly Detection")
ecg_index = st.sidebar.slider("Select ECG Index", 0, len(n_test_data) - 1, 0)
use_lime = st.sidebar.button("Show LIME Explanation")

# Make predictions and calculate threshold
if df is not None:
    reconstructed = autoencoder(n_train_data)
    train_loss = losses.mae(reconstructed, n_train_data)
    threshold = np.mean(train_loss) + 2 * np.std(train_loss)

    def prediction(model, data, threshold):
        rec = model(data)
        loss = losses.mae(rec, data)
        return tf.math.less(loss, threshold)

    plot_residual_with_lime(n_test_data, ecg_index)
