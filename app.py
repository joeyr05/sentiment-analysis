import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)

# Download required NLTK resources if not already downloaded
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

download_nltk_resources()

# Create tokenizer and lemmatizer
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    try:
        if not isinstance(text, str):
            text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

        # Remove mentions and hashtags
        text = re.sub(r'\@\w+|\#', '', text)

        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', '', text)

        # Tokenize
        tokens = tokenizer.tokenize(text)

        # Remove numbers and short words
        tokens = [token for token in tokens if not token.isdigit() and len(token) > 2]

        # Lemmatization
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        return " ".join(tokens)
    except Exception as e:
        st.error(f"Error processing text: {e}")
        return ""

# Train or load model
@st.cache_resource
def get_model():
    model_path = "sentiment_model.pkl"
    
    # Check if model already exists
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading saved model: {e}")
    
    # If model doesn't exist or couldn't be loaded, train a new one
    try:
        st.info("Training new model. This may take a few minutes...")
        
        # Check if the CSV files exist
        train_file_exists = os.path.exists('twitter_training.csv')
        test_file_exists = os.path.exists('twitter_validation.csv')
        
        if not train_file_exists or not test_file_exists:
            st.warning("Training or validation data files not found. Please ensure 'twitter_training.csv' and 'twitter_validation.csv' are in the same directory as this app.")
            # Create a minimal training dataset as fallback
            st.warning("Using minimal example dataset as fallback.")
            data = {
                "tweet": ["I love this product!", "This is the worst experience ever!", 
                         "It's okay, nothing special.", "Amazing service!", "Terrible quality.",
                         "Neutral opinion on this.", "Great experience!", "Awful customer service.",
                         "Neither good nor bad.", "Best purchase ever!"],
                "sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative", 
                             "Neutral", "Positive", "Negative", "Neutral", "Positive"]
            }
            train_df = pd.DataFrame(data)
            test_df = train_df.copy()  # Use same data for testing in fallback scenario
        else:
            # Load the datasets
            st.info("Loading datasets...")
            train_df = pd.read_csv('twitter_training.csv')
            test_df = pd.read_csv('twitter_validation.csv')
        
        # Keep necessary columns
        st.info("Processing dataframes...")
        train_df = train_df[["tweet", "sentiment"]]
        test_df = test_df[["tweet", "sentiment"]]
        
        # Apply preprocessing
        st.info("Preprocessing tweets...")
        train_df["clean_text"] = train_df["tweet"].apply(preprocess_text)
        test_df["clean_text"] = test_df["tweet"].apply(preprocess_text)
        
        # Convert labels
        st.info("Converting sentiment labels...")
        label_mapping = {"Positive": 1, "Negative": 0, "Neutral": 2}
        train_df["sentiment"] = train_df["sentiment"].map(label_mapping)
        test_df["sentiment"] = test_df["sentiment"].map(label_mapping)
        
        # Remove NaN values
        train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                strip_accents='unicode',
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            )),
            ('clf', MultinomialNB())
        ])
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'clf__alpha': [0.1, 0.5, 1.0, 1.5, 2.0],  # Smoothing parameter
        }
        
        # Perform grid search
        st.info("Performing grid search...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(train_df["clean_text"], train_df["sentiment"])
        
        # Print best parameters
        best_params = grid_search.best_params_
        st.info(f"Best parameters: {best_params}")
        
        # Evaluate on validation set
        st.info("Evaluating model...")
        y_pred = grid_search.predict(test_df["clean_text"])
        
        # Calculate metrics
        accuracy = accuracy_score(test_df['sentiment'], y_pred)
        report = classification_report(test_df['sentiment'], y_pred, output_dict=True)
        
        # Store metrics in the model object for display
        grid_search.metrics = {
            'accuracy': accuracy,
            'report': report
        }
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(grid_search, f)
            
        return grid_search
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        # Return a basic fallback model
        fallback_model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        fallback_model.metrics = {'accuracy': 0, 'report': {}}
        return fallback_model

# Prediction function
def predict_sentiment(tweet, model):
    try:
        processed_tweet = preprocess_text(tweet)
        prediction = model.predict([processed_tweet])[0]
        sentiment_map = {1: "Positive", 0: "Negative", 2: "Neutral"}
        confidence = model.predict_proba([processed_tweet])[0].max()
        return sentiment_map[prediction], confidence, processed_tweet
    except Exception as e:
        st.error(f"Error predicting sentiment: {e}")
        return "Error", 0.0, ""

# Display model metrics
def show_model_metrics(model):
    try:
        if hasattr(model, 'metrics'):
            metrics = model.metrics
            st.subheader("Model Performance")
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            
            if 'report' in metrics and metrics['report']:
                report = metrics['report']
                # Create a DataFrame for better display
                if '0' in report and '1' in report and '2' in report:
                    df_report = pd.DataFrame({
                        'Metric': ['Precision', 'Recall', 'F1-Score'],
                        'Negative': [report['0']['precision'], report['0']['recall'], report['0']['f1-score']],
                        'Positive': [report['1']['precision'], report['1']['recall'], report['1']['f1-score']],
                        'Neutral': [report['2']['precision'], report['2']['recall'], report['2']['f1-score']]
                    })
                    st.dataframe(df_report)
    except Exception as e:
        st.warning(f"Could not display metrics: {e}")


# Main app function
def main():
    st.title("Tweet Sentiment Analyzer")
    st.markdown("""
    This app analyzes the sentiment of tweets as positive, negative, or neutral using a Naive Bayes classifier.
    Enter your tweet below to analyze its sentiment!
    """)
    
    # Get the model (this will train it if necessary)
    with st.spinner("Loading model..."):
        model = get_model()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Analyze Tweets", "Model Information"])
    
    with tab1:
        # Input for tweet
        tweet_input = st.text_area("Enter a tweet:", height=100, max_chars=280, 
                                 placeholder="Type your tweet here...")
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("Analyze Sentiment", type="primary")
        
        # Display results
        if analyze_button and tweet_input:
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence, processed_tweet = predict_sentiment(tweet_input, model)
            
            # Result container
            st.markdown("### Analysis Result")
            
            # Display sentiment with appropriate color
            if sentiment == "Positive":
                sentiment_color = "green"
                emoji = "😃"
            elif sentiment == "Negative":
                sentiment_color = "red"
                emoji = "😞"
            else:
                sentiment_color = "gray"
                emoji = "😐"
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<h2 style='color:{sentiment_color};'>{sentiment} {emoji}</h2>", unsafe_allow_html=True)
            with col2:
                # Confidence gauge
                st.markdown("#### Confidence")
                st.progress(confidence)
                st.text(f"{confidence:.2%}")
            
            # Show processed tweet
            with st.expander("See processed tweet"):
                st.text(processed_tweet)
        
        # Show examples
        with st.expander("Try with examples"):
            example_tweets = [
                "I love this product!",
                "This is the worst experience ever!",
                "It's okay, nothing special.",
                "I am getting on borderlands and I will murder you all",
                "I am coming to the borders and I will kill you all"
            ]
            
            for i, example in enumerate(example_tweets):
                if st.button(f"Example {i+1}", key=f"example_{i}"):
                    # Use session state to store the selected example
                    if 'tweet_input' not in st.session_state:
                        st.session_state['tweet_input'] = example
                        st.rerun()
    
    with tab2:
        st.subheader("Model Information")
        st.markdown("""
        This sentiment analyzer is trained using a Naive Bayes classifier with TF-IDF vectorization.
        The model is trained on the `twitter_training.csv` dataset and validated using `twitter_validation.csv`.
        """)
        
        # Show model parameters if available
        if hasattr(model, 'best_params_'):
            st.json(model.best_params_)
        
        # Show model metrics
        show_model_metrics(model)
        
        # Data preprocessing explanation
        with st.expander("Data Preprocessing"):
            st.markdown("""
            The tweets undergo the following preprocessing steps:
            1. Convert to lowercase
            2. Remove URLs, mentions, and hashtags
            3. Remove special characters and emojis
            4. Remove stopwords and short words
            5. Lemmatize words to their base form
            
            This helps in reducing noise and improving model accuracy.
            """)

# Add a sidebar with information
with st.sidebar:
    st.title("About")
    st.info("""
    This sentiment analyzer uses Natural Language Processing and Machine Learning
    to determine if a tweet has a positive, negative, or neutral sentiment.
    
    The model uses:
    - Text preprocessing
    - TF-IDF vectorization
    - Naive Bayes classification
    - Grid search for hyperparameter tuning
    """)
    
    st.divider()
    
    st.subheader("How to use")
    st.markdown("""
    1. Enter your tweet in the text area
    2. Click "Analyze Sentiment"
    3. View the sentiment prediction and confidence score
    
    You can also try the example tweets to see how the model performs.
    """)
    
    st.divider()
    
    st.caption("🔄 Model is automatically trained if not found")

# Run the main app
if __name__ == "__main__":
    main()