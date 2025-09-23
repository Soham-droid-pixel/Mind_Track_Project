"""
MindTrack: Mental Health Sentiment Analyzer
Main Streamlit application for analyzing mental health risk in social media posts.
"""

import streamlit as st
import praw
import time
from datetime import datetime
import pandas as pd
import sys
import os

# Add the current directory to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils import (
    load_model_and_tokenizer,
    predict_single_text,
    create_lime_explainer,
    explain_text,
    predictor,
    preprocess_text,
    get_model_info
)

# Configure Streamlit page
st.set_page_config(
    page_title="MindTrack - Mental Health Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.risk-high {
    color: #ff4444;
    font-weight: bold;
}

.risk-normal {
    color: #00aa00;
    font-weight: bold;
}

.confidence-score {
    font-size: 1.2rem;
    margin: 1rem 0;
}

.post-container {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #f9f9f9;
}

.timestamp {
    color: #666;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load and cache the model and tokenizer."""
    try:
        model, tokenizer = load_model_and_tokenizer()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_resource
def get_lime_explainer():
    """Create and cache LIME explainer."""
    return create_lime_explainer()

def display_prediction_result(text, prediction_label, confidence, show_explanation=True):
    """Display prediction results with styling."""
    
    # Display prediction
    if prediction_label == "Risk Detected":
        st.markdown(f'<p class="risk-high">⚠️ {prediction_label}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="risk-normal">✅ {prediction_label}</p>', unsafe_allow_html=True)
    
    # Display confidence score
    st.markdown(f'<p class="confidence-score">Confidence: {confidence:.2%}</p>', unsafe_allow_html=True)
    
    # Show explanation if requested
    if show_explanation:
        with st.expander("🔍 View Explanation (LIME Analysis)", expanded=False):
            try:
                explainer = get_lime_explainer()
                explanation = explain_text(explainer, text, predictor, num_features=10)
                
                if explanation:
                    # Display explanation as HTML
                    explanation_html = explanation.as_html()
                    st.markdown(explanation_html, unsafe_allow_html=True)
                    
                    # Show feature importance
                    st.subheader("Word Importance Scores")
                    exp_list = explanation.as_list(label=1)  # Risk class
                    
                    if exp_list:
                        df_exp = pd.DataFrame(exp_list, columns=['Word', 'Importance'])
                        df_exp = df_exp.sort_values('Importance', key=abs, ascending=False)
                        st.dataframe(df_exp, use_container_width=True)
                    else:
                        st.info("No significant features found for explanation.")
                else:
                    st.error("Unable to generate explanation.")
                    
            except Exception as e:
                st.error(f"Error generating explanation: {str(e)}")

def single_post_analyzer():
    """Single post analysis interface."""
    
    st.markdown('<h1 class="main-header">🧠 MindTrack: Analyze a Single Post</h1>', unsafe_allow_html=True)
    
    # Load model
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("❌ Model not loaded. Please check your model files.")
        return
    
    # Display model info
    with st.expander("ℹ️ Model Information", expanded=False):
        model_info = get_model_info()
        for key, value in model_info.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    # Input text area
    st.subheader("📝 Enter text to analyze:")
    user_input = st.text_area(
        "Type or paste the social media post you want to analyze:",
        height=150,
        placeholder="Example: I'm feeling really down lately and nothing seems to matter anymore..."
    )
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        show_explanation = st.checkbox("Show LIME Explanation", value=True)
    with col2:
        preprocess_input = st.checkbox("Preprocess Text", value=True)
    
    # Analyze button
    if st.button("🔍 Analyze Post", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing text..."):
                try:
                    # Preprocess if requested
                    text_to_analyze = preprocess_text(user_input) if preprocess_input else user_input
                    
                    # Get prediction
                    prediction_label, confidence, probabilities = predict_single_text(
                        text_to_analyze, model, tokenizer
                    )
                    
                    # Display results
                    st.subheader("📊 Analysis Results:")
                    display_prediction_result(text_to_analyze, prediction_label, confidence, show_explanation)
                    
                    # Show probability breakdown
                    with st.expander("📈 Probability Breakdown", expanded=False):
                        prob_df = pd.DataFrame({
                            'Class': ['Normal', 'Risk'],
                            'Probability': probabilities
                        })
                        st.bar_chart(prob_df.set_index('Class'))
                        st.dataframe(prob_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

def live_subreddit_monitor():
    """Live subreddit monitoring interface."""
    
    st.markdown('<h1 class="main-header">📡 MindTrack: Live Subreddit Monitor</h1>', unsafe_allow_html=True)
    
    # Load model
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("❌ Model not loaded. Please check your model files.")
        return
    
    # Reddit API configuration
    st.subheader("🔧 Reddit API Configuration")
    
    # Use secrets if available, otherwise input fields
    if 'reddit' in st.secrets:
        client_id = st.secrets.reddit.client_id
        client_secret = st.secrets.reddit.client_secret
        user_agent = st.secrets.reddit.user_agent
        st.success("✅ Using Reddit API credentials from secrets.")
    else:
        st.info("💡 Add your Reddit API credentials to secrets.toml or enter them below.")
        col1, col2 = st.columns(2)
        with col1:
            client_id = st.text_input("Client ID", type="password")
            user_agent = st.text_input("User Agent", value="MindTrack:v1.0")
        with col2:
            client_secret = st.text_input("Client Secret", type="password")
    
    # Subreddit configuration
    st.subheader("📱 Subreddit Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        subreddit_name = st.text_input("Subreddit Name", value="mentalhealthsupport")
    with col2:
        max_posts = st.number_input("Max Posts to Display", min_value=5, max_value=50, value=10)
    with col3:
        update_interval = st.number_input("Update Interval (seconds)", min_value=5, max_value=60, value=10)
    
    # Monitoring controls
    col1, col2 = st.columns(2)
    with col1:
        start_monitoring = st.button("🚀 Start Monitoring", type="primary")
    with col2:
        stop_monitoring = st.button("⏹️ Stop Monitoring")
    
    # Initialize session state
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    if 'posts_data' not in st.session_state:
        st.session_state.posts_data = []
    
    if start_monitoring:
        st.session_state.monitoring = True
    if stop_monitoring:
        st.session_state.monitoring = False
    
    # Monitoring logic
    if st.session_state.monitoring:
        if not all([client_id, client_secret, user_agent]):
            st.error("❌ Please provide all Reddit API credentials.")
            st.session_state.monitoring = False
            return
        
        try:
            # Initialize Reddit API
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            st.success(f"✅ Connected to Reddit. Monitoring r/{subreddit_name}")
            
            # Create placeholder for live feed
            placeholder = st.empty()
            
            # Monitor subreddit
            subreddit = reddit.subreddit(subreddit_name)
            
            for submission in subreddit.stream.submissions(skip_existing=True):
                if not st.session_state.monitoring:
                    break
                
                try:
                    # Analyze the post
                    post_text = f"{submission.title} {submission.selftext}"
                    if len(post_text.strip()) > 10:  # Only analyze substantial posts
                        
                        prediction_label, confidence, probabilities = predict_single_text(
                            post_text, model, tokenizer
                        )
                        
                        # Add to posts data
                        post_data = {
                            'timestamp': datetime.now(),
                            'title': submission.title[:100] + "..." if len(submission.title) > 100 else submission.title,
                            'author': str(submission.author) if submission.author else "Unknown",
                            'prediction': prediction_label,
                            'confidence': confidence,
                            'url': f"https://reddit.com{submission.permalink}"
                        }
                        
                        st.session_state.posts_data.insert(0, post_data)
                        
                        # Keep only the latest posts
                        if len(st.session_state.posts_data) > max_posts:
                            st.session_state.posts_data = st.session_state.posts_data[:max_posts]
                        
                        # Update display
                        with placeholder.container():
                            st.subheader("📊 Live Feed")
                            
                            for i, post in enumerate(st.session_state.posts_data):
                                with st.container():
                                    st.markdown(f"""
                                    <div class="post-container">
                                        <div class="timestamp">{post['timestamp'].strftime('%H:%M:%S')} - u/{post['author']}</div>
                                        <h4>{post['title']}</h4>
                                        <p class="{'risk-high' if post['prediction'] == 'Risk Detected' else 'risk-normal'}">
                                            {post['prediction']} (Confidence: {post['confidence']:.1%})
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        time.sleep(1)  # Brief pause between posts
                
                except Exception as e:
                    st.error(f"Error processing post: {str(e)}")
                    continue
                
        except Exception as e:
            st.error(f"Error connecting to Reddit: {str(e)}")
            st.session_state.monitoring = False
    
    # Display current posts if any
    if st.session_state.posts_data and not st.session_state.monitoring:
        st.subheader("📊 Recent Posts")
        for post in st.session_state.posts_data:
            with st.container():
                st.markdown(f"""
                <div class="post-container">
                    <div class="timestamp">{post['timestamp'].strftime('%H:%M:%S')} - u/{post['author']}</div>
                    <h4>{post['title']}</h4>
                    <p class="{'risk-high' if post['prediction'] == 'Risk Detected' else 'risk-normal'}">
                        {post['prediction']} (Confidence: {post['confidence']:.1%})
                    </p>
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Sidebar navigation
    st.sidebar.title("🧠 MindTrack")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose Analysis Mode:",
        ["Single Post Analyzer", "Live Subreddit Monitor"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About MindTrack
    
    MindTrack is an AI-powered tool for analyzing mental health sentiment in social media posts.
    
    **Features:**
    - 🔍 Single post analysis with explanations
    - 📡 Live subreddit monitoring
    - 🎯 LIME explainability
    - 📊 Confidence scoring
    
    **Disclaimer:** This tool is for educational purposes only and should not be used as a substitute for professional mental health advice.
    """)
    
    # Route to appropriate page
    if page == "Single Post Analyzer":
        single_post_analyzer()
    elif page == "Live Subreddit Monitor":
        live_subreddit_monitor()

if __name__ == "__main__":
    main()