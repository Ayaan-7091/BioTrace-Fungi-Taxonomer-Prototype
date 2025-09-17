import streamlit as st
import pandas as pd
import numpy as np
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib # To save and load the LabelEncoder

# --- Page Configuration ---
st.set_page_config(
    page_title="BioTrace Dashboard",
    page_icon="🧬",
    layout="wide"
)

# --- Helper Functions ---

# This function preprocesses a single DNA sequence for the model
def one_hot_encode_single(sequence, max_len=600):
    """Preprocesses a single DNA sequence string."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    # Create a 3D array with shape (1, max_len, num_features)
    encoded = np.zeros((1, max_len, len(mapping)))
    padded_seq = sequence[:max_len].upper().ljust(max_len, 'N')
    for i, base in enumerate(padded_seq):
        if base in mapping:
            encoded[0, i, mapping[base]] = 1
    return encoded

# Loading here the trained model once and caching it for performance
@st.cache_resource
def load_prediction_model():
    """Loads the pre-trained Keras model."""
    try:
        model = load_model('fungi_family_classifier.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model 'fungi_family_classifier.h5': {e}")
        return None

# Create and fit the LabelEncoder from the original data
@st.cache_resource
def get_label_encoder(data_path='fungi_labeled_data.csv'):
    """Creates, fits, and returns a LabelEncoder for the family names."""
    try:
        df = pd.read_csv(data_path)
        df.dropna(subset=['family'], inplace=True)
        df = df[df['family'] != 'Unknown_Family'].copy()
        
        family_counts = df['family'].value_counts()
        families_to_keep = family_counts[family_counts >= 2].index.tolist()
        df_filtered = df[df['family'].isin(families_to_keep)].copy()

        encoder = LabelEncoder()
        encoder.fit(df_filtered['family'])
        return encoder
    except FileNotFoundError:
        st.error(f"Error: The data file '{data_path}' is needed to map predictions to family names.")
        return None

# --- Main App UI ---

st.title("🧬 BioTrace: AI-Powered eDNA Discovery Engine")
st.write("An AI-driven pipeline to classify, annotate, and discover biodiversity from environmental DNA datasets.")

# --- Tabbed Interface for Clarity ---
tab1, tab2 = st.tabs(["Analyze Full Sample", "Analyze Single Sequence"])


# --- TAB 1: Full Sample Analysis ---
with tab1:
    st.header("Analyze a Full eDNA Sample")
    st.write("This feature simulates the analysis of a complete eDNA sample, showing the final outputs of the BioTrace pipeline.")

    # We use a button to trigger the display of pre-computed results
    if st.button("🚀 Run Full Analysis"):
        try:
            with st.spinner('AI is analyzing genetic fingerprints and discovering clusters... Please wait.'):
                time.sleep(3) # Simulate a long process
                
                # Load the pre-computed results
                results_df = pd.read_csv('fungi_final_clusters.csv')
                num_clusters = len(results_df['cluster'].unique()) - (1 if -1 in results_df['cluster'].unique() else 0)
                novel_taxa_count = results_df['cluster'].value_counts().loc[lambda x : x < 3].count()

            st.success('Analysis Complete!')
            
            # --- Display Results ---
            st.subheader("🔬 Analysis Results")

            # --- Summary Metrics ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Sequences Analyzed", len(results_df), help="Total number of valid DNA sequences processed.")
            col2.metric("Distinct Species/OTUs Discovered", num_clusters, help="Number of unique genetic groups found by the AI.")
            col3.metric("Potentially Novel/Rare Taxa", novel_taxa_count, help="Number of small clusters (fewer than 3 members), indicating rare or potentially new species.")

            # --- Cluster Visualization ---
            st.subheader("Genetic Cluster Map")
            st.image('cluster_visualization.png', caption='Each color represents a unique species cluster discovered by the AI. Proximity indicates genetic similarity.')
            
            # --- Interactive Data Table ---
            st.subheader("Explore the Discovered Clusters")
            st.dataframe(results_df)

        except FileNotFoundError as e:
            st.error(f"Error: Missing result file. Please make sure '{e.filename}' is in the same folder as the dashboard.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


# --- TAB 2: Single Sequence Analysis ---
with tab2:
    st.header("🔍 Classify a Single DNA Sequence")
    sequence_input = st.text_area("Paste a raw DNA sequence here:", height=150, placeholder="Example: TATCTGGTTGATCCTGCCAGTAGTCATATGCTTGTCTCAAAGATTAAGCCATGCATGT...")

    if st.button("Classify Sequence"):
        if sequence_input:
            # Load model and encoder
            model = load_prediction_model()
            label_encoder = get_label_encoder()

            if model and label_encoder:
                with st.spinner("Running sequence through the AI model..."):
                    # Preprocess the input sequence
                    processed_sequence = one_hot_encode_single(sequence_input)
                    
                    # Make a prediction
                    prediction = model.predict(processed_sequence)
                    
                    # Get the top prediction details
                    top_class_index = np.argmax(prediction)
                    top_confidence = np.max(prediction)
                    family_name = label_encoder.inverse_transform([top_class_index])[0]
                
                st.success("Classification Complete!")
                st.subheader("Prediction Results:")
                
                col1, col2 = st.columns(2)
                col1.metric(label="Most Likely Family", value=family_name)
                col2.metric(label="Confidence", value=f"{top_confidence * 100:.2f}%")
        else:
            st.warning("Please paste a DNA sequence to analyze.")