import streamlit as st
import os
from sourcecode.model import FakeNewsDetector
from sourcecode.utils import load_model_components, predict_news

def main():
    """
    Main Streamlit application for Fake News Detection
    """
    st.set_page_config(
        page_title="Fake News Detection",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 Fake News Detection System")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox(
        "Choose an option:",
        ["Predict News", "Train Model", "About"]
    )
    
    if option == "Predict News":
        predict_page()
    elif option == "Train Model":
        train_page()

def predict_page():
    st.header("News Prediction")

    model_path = "model/fake_news_detection_model.pkl"
    vectorizer_path = "vectorizer/tfidf_vectorizer.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        st.error("⚠️ Model not found! Please train the model first.")
        st.info("Go to 'Train Model' section to train a new model.")
        return
    
    # Load model
    try:
        model, vectorizer = load_model_components(model_path, vectorizer_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Text input
    st.subheader("Enter News Text")
    news_text = st.text_area(
        "Paste the news article here:",
        height=200,
        placeholder="Enter the news article you want to verify..."
    )
    
    if st.button("🚀 Analyze News", type="primary"):
        if news_text.strip():
            with st.spinner("Analyzing..."):
                try:
                    prediction, confidence = predict_news(news_text, model, vectorizer)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == "Real News":
                            st.success(f"**{prediction}**")
                        else:
                            st.error(f"**{prediction}**")
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Additional info
                    st.info(f"The model is {confidence:.1%} confident in this prediction.")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        else:
            st.warning("Please enter some text to analyze.")

def train_page():
    st.header("🎯 Model Training")
    # Check if data files exist
    fake_path = "data/Fake.csv"
    true_path = "data/True.csv"
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(fake_path):
            st.success("Fake.csv found")
        else:
            st.error("Fake.csv not found")
    
    with col2:
        if os.path.exists(true_path):
            st.success("True.csv found")
        else:
            st.error("True.csv not found")
    
    if os.path.exists(fake_path) and os.path.exists(true_path):
        st.subheader("Training Configuration")
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            max_df = st.slider("Max Document Frequency", 0.1, 1.0, 0.7)
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        
        with col2:
            random_state = st.number_input("Random State", value=42)
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Initialize detector with custom parameters
                    detector = FakeNewsDetector(
                        max_df=max_df,
                        test_size=test_size,
                        random_state=int(random_state)
                    )
                    
                    # Train the model
                    metrics = detector.train_complete_pipeline(fake_path, true_path)
                    
                    # Display results
                    st.success("🎉 Model training completed!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    
                    with col2:
                        st.metric("Dataset Size", len(detector.df))
                    
                    # Show classification report
                    st.subheader("📊 Model Performance")
                    st.text(metrics['classification_report'])
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()