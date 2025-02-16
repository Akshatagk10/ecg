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

# Error handling for data loading
@st.cache_data
def load_data(file=None):
    try:
        if file is not None:
            df = pd.read_csv(file)
        else:
            df = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Error handling for model training
@st.cache_resource
def load_model():
    try:
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
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load dataset or allow user to upload
uploaded_file = st.sidebar.file_uploader("Upload your ECG data (CSV)", type=["csv"])

# Load ECG data
df = load_data(uploaded_file)

if df is not None:
    # Prepare data
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    # Split data into train and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)

    # Normalize the data
    min_val, max_val = tf.reduce_min(train_data), tf.reduce_max(train_data)
    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data, test_data = map(lambda x: tf.cast(x, dtype=tf.float32), [train_data, test_data])

    # Convert labels to boolean
    train_labels, test_labels = train_labels.astype(bool), test_labels.astype(bool)

    # Separate normal and abnormal ECG data
    n_train_data, n_test_data = train_data[train_labels], test_data[test_labels]
    an_train_data, an_test_data = train_data[~train_labels], test_data[~test_labels]

    # Load and train model
    autoencoder = load_model()
    if autoencoder:
        try:
            autoencoder.fit(n_train_data, n_train_data, epochs=20, batch_size=512, validation_data=(n_test_data, n_test_data))
        except Exception as e:
            st.error(f"Error during model training: {e}")
            autoencoder = None
else:
    st.warning("No ECG data available. Please upload a dataset.")

# Define LIME Explainer
explainer = LimeTabularExplainer(
    training_data=n_train_data.numpy(),
    mode="regression",
    feature_names=[f"Feature {i}" for i in range(n_train_data.shape[1])],
    discretize_continuous=True
)

# Function to plot original vs reconstructed ECG
def plot(data, index):
    fig, ax = plt.subplots()
    enc_img = autoencoder.encoder(data)
    dec_img = autoencoder.decoder(enc_img)
    
    ax.plot(data[index], 'b', label='Input')
    ax.plot(dec_img[index], 'r', label='Reconstruction')
    ax.fill_between(np.arange(140), data[index], dec_img[index], color='lightcoral', alpha=0.5, label='Error')
    ax.legend()
    st.pyplot(fig)

# Function to plot LIME explanations
def plot_with_lime(data, index):
    fig, ax = plt.subplots()
    enc_img = autoencoder.encoder(data)
    dec_img = autoencoder.decoder(enc_img)

    ax.plot(data[index], 'b', label='Input')
    ax.plot(dec_img[index], 'r', label='Reconstruction')
    ax.fill_between(np.arange(140), data[index], dec_img[index], color='lightcoral', alpha=0.5, label='Error')
    ax.legend()

    # Generate LIME explanation
    exp = explainer.explain_instance(data[index].numpy(), autoencoder.predict)
    lime_fig = exp.as_pyplot_figure()

    # Display both plots in Streamlit
    st.pyplot(fig)
    st.pyplot(lime_fig)

# Sidebar inputs
st.sidebar.title("ECG Anomaly Detection")
ecg_type = st.sidebar.selectbox("Select ECG Type", ["Normal ECG", "Abnormal ECG"])
ecg_index = st.sidebar.slider("Select ECG Index", 0, len(n_test_data) - 1, 0)
use_lime = st.sidebar.checkbox("Show LIME Explanation", False)

# Make predictions and calculate anomaly detection threshold
if autoencoder:
    reconstructed = autoencoder(n_train_data)
    train_loss = losses.mae(reconstructed, n_train_data)
    threshold = np.mean(train_loss) + 2 * np.std(train_loss)

    # Prediction function
    def prediction(model, data, threshold):
        rec = model(data)
        loss = losses.mae(rec, data)
        return tf.math.less(loss, threshold)

    # Display ECG plots
    if use_lime:
        plot_with_lime(n_test_data, ecg_index)
    else:
        plot(n_test_data, ecg_index)

    # Display anomaly detection results
    if st.sidebar.button("Make Predictions"):
        pred = prediction(autoencoder, n_test_data, threshold)
        acc = np.sum(pred.numpy()) / len(pred.numpy()) * 100
        st.sidebar.write(f"Anomaly Detection Accuracy: {acc:.2f}%")
