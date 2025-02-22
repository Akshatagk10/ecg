import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
from tensorflow.keras import layers, losses, Model
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

# Suppress warnings
warnings.filterwarnings('ignore')

# Set Streamlit page configuration
st.set_page_config(page_title="ECG Anomaly Detection", page_icon="💓", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        .main { background-color: #f7f9fc; }
        .sidebar .sidebar-content { background-color: #e8f0fe; }
        h1 { color: #003366; }
    </style>
""", unsafe_allow_html=True)

# Load Data Function
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

# Define Autoencoder Model
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

# Sidebar - File Uploader
st.sidebar.title("Upload ECG Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Load Data
with st.spinner("Loading Data..."):
    df = load_data(uploaded_file)

if df is not None:
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)
    min_val, max_val = np.min(train_data), np.max(train_data)
    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)
    train_data, test_data = map(lambda x: tf.cast(x, dtype=tf.float32), [train_data, test_data])
    train_labels, test_labels = train_labels.astype(bool), test_labels.astype(bool)
    n_train_data, n_test_data = train_data[train_labels], test_data[test_labels]

    # Load and Train Model
    autoencoder = load_model()
    with st.spinner("Training Model..."):
        autoencoder.fit(n_train_data, n_train_data, epochs=20, batch_size=64, validation_data=(test_data, test_data))

    # Define LIME Explainer
    explainer = LimeTabularExplainer(
        training_data=n_train_data.numpy(),
        mode="regression",
        feature_names=[f"Feature {i}" for i in range(n_train_data.shape[1])],
        discretize_continuous=True
    )

    # Sidebar - User Controls
    st.sidebar.title("ECG Analysis")
    ecg_index = st.sidebar.slider("Select ECG Index", 0, len(n_test_data) - 1, 0)
    use_lime = st.sidebar.checkbox("Show LIME Explanation", False)

    # Model Predictions & Threshold
    reconstructed = autoencoder(n_train_data)
    train_loss = losses.mae(reconstructed, n_train_data)
    threshold = np.mean(train_loss) + 2 * np.std(train_loss)

    def prediction(model, data, threshold):
        rec = model(data)
        loss = losses.mae(rec, data)
        return tf.math.less(loss, threshold)

    # Function to Plot ECG Data
    def plot_ecg(data, index, show_lime=False):
        enc_img = autoencoder.encoder(data)
        dec_img = autoencoder.decoder(enc_img)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data[index], mode='lines', name='Input', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=dec_img[index], mode='lines', name='Reconstruction', line=dict(color='red')))
        fig.add_trace(go.Scatter(y=(data[index] - dec_img[index]), mode='lines', fill='tozeroy', name='Error', line=dict(color='lightcoral')))
        
        st.plotly_chart(fig, use_container_width=True)
        
        if show_lime:
            exp = explainer.explain_instance(data[index].numpy(), lambda x: autoencoder.predict(x))
            lime_fig = exp.as_pyplot_figure()
            st.pyplot(lime_fig)

    # Display ECG Visualization
    st.subheader("ECG Signal Analysis")
    plot_ecg(n_test_data, ecg_index, use_lime)

else:
    st.warning("No ECG data available. Please upload a dataset.")
