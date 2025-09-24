"""
MindTrack: Mental Health Sentiment Analyzer
Enhanced Streamlit application with improved UI and performance.
"""

import streamlit as st
import praw
import time
from datetime import datetime
import pandas as pd
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

# Add the current directory to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils import (
    load_model_and_tokenizer,
    predict_single_text,
    create_lime_explainer,
    explain_text,
    predictor,
    format_lime_explanation
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
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.risk-high {
    color: #ff4444;
    font-weight: bold;
    font-size: 1.3em;
    background-color: #ffe6e6;
    padding: 10px;
    border-radius: 8px;
    border-left: 5px solid #ff4444;
}

.risk-normal {
    color: #44aa44;
    font-weight: bold;
    font-size: 1.3em;
    background-color: #e6ffe6;
    padding: 10px;
    border-radius: 8px;
    border-left: 5px solid #44aa44;
}

.confidence-score {
    font-size: 1.2em;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}

.confidence-high {
    background-color: #e8f5e8;
    border: 2px solid #4caf50;
}

.confidence-medium {
    background-color: #fff8e1;
    border: 2px solid #ff9800;
}

.confidence-low {
    background-color: #ffebee;
    border: 2px solid #f44336;
}

.explanation-box {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #dee2e6;
    margin: 15px 0;
}

.word-tag {
    display: inline-block;
    padding: 5px 12px;
    margin: 3px;
    border-radius: 20px;
    font-size: 0.9em;
    font-weight: 500;
}

.positive-influence {
    background-color: #ffcdd2;
    color: #c62828;
    border: 1px solid #f44336;
}

.negative-influence {
    background-color: #c8e6c9;
    color: #2e7d32;
    border: 1px solid #4caf50;
}

.quick-examples {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 20px 0;
}

.example-button {
    background-color: #e3f2fd;
    border: 1px solid #2196f3;
    border-radius: 20px;
    padding: 8px 16px;
    margin: 5px;
    cursor: pointer;
    font-size: 0.9em;
    color: #1976d2;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.info-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    border-left: 4px solid #1f77b4;
    margin: 15px 0;
}

.stButton > button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
    font-weight: 600;
    background: linear-gradient(45deg, #1f77b4, #17a2b8);
    color: white;
    border: none;
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(31,119,180,0.3);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'lime_explainer' not in st.session_state:
    st.session_state.lime_explainer = None

@st.cache_resource
def load_model_cached():
    """Load model with Streamlit caching for better performance."""
    try:
        model, tokenizer = load_model_and_tokenizer('./saved_model')
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

@st.cache_resource
def load_lime_explainer_cached():
    """Load LIME explainer with caching."""
    try:
        return create_lime_explainer()
    except Exception as e:
        st.error(f"Failed to load LIME explainer: {str(e)}")
        return None

def display_prediction_results(text, prediction_label, confidence, explanation_data=None):
    """Display prediction results with enhanced styling."""
    
    # Main prediction result
    if prediction_label == "Risk Detected":
        st.markdown(f'<div class="risk-high">🚨 {prediction_label}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-normal">✅ {prediction_label}</div>', unsafe_allow_html=True)
    
    # Confidence score with color coding
    confidence_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"
    confidence_emoji = "🎯" if confidence > 0.8 else "📊" if confidence > 0.6 else "⚠️"
    
    st.markdown(f'''
    <div class="confidence-score {confidence_class}">
        {confidence_emoji} Confidence Score: {confidence:.1%}
    </div>
    ''', unsafe_allow_html=True)
    
    # Display explanation if available
    if explanation_data:
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        st.markdown("### 🔍 AI Explanation")
        
        # Summary
        st.markdown(f"**{explanation_data['explanation_summary']}**")
        
        # Positive influences
        if explanation_data['positive_influences']:
            st.markdown("**Words that increased the prediction:**")
            words_html = ""
            for word_data in explanation_data['positive_influences']:
                words_html += f'<span class="word-tag positive-influence" title="{word_data["description"]}">{word_data["word"]} ({word_data["importance"]:.2f})</span>'
            st.markdown(words_html, unsafe_allow_html=True)
        
        # Negative influences
        if explanation_data['negative_influences']:
            st.markdown("**Words that decreased the prediction:**")
            words_html = ""
            for word_data in explanation_data['negative_influences']:
                words_html += f'<span class="word-tag negative-influence" title="{word_data["description"]}">{word_data["word"]} ({word_data["importance"]:.2f})</span>'
            st.markdown(words_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def single_text_analysis_tab():
    """Enhanced single text analysis tab."""
    st.markdown('<h2 style="text-align: center;">📝 Single Text Analysis</h2>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model, tokenizer = load_model_cached()
    
    if model is None or tokenizer is None:
        st.error("❌ Model not loaded. Please check your model files.")
        return
    
    # Quick examples
    st.markdown("""
    <div class="quick-examples">
        <h4>💡 Try these examples:</h4>
        <p>Click on any example to test it quickly!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("😊 Positive Example"):
            st.session_state.example_text = "I had an amazing day today! Feeling grateful for all the good things in my life."
        if st.button("😔 Concerning Example"):
            st.session_state.example_text = "I don't see the point in anything anymore. Everything feels hopeless."
    
    with col2:
        if st.button("😐 Neutral Example"):
            st.session_state.example_text = "Just finished my work for today. Going to grab some dinner now."
        if st.button("🤔 Mixed Example"):
            st.session_state.example_text = "Having a tough time lately but trying to stay positive and reach out for support."
    
    # Text input
    input_text = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.get('example_text', ''),
        height=120,
        placeholder="Type or paste any text here to analyze its mental health sentiment..."
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Text", type="primary")
    
    if analyze_button and input_text.strip():
        with st.spinner("🤖 Analyzing text..."):
            # Get prediction
            prediction_label, confidence, probabilities = predict_single_text(input_text, model, tokenizer)
            
            # Get LIME explanation
            lime_explainer = load_lime_explainer_cached()
            explanation = None
            explanation_data = None
            
            if lime_explainer:
                with st.spinner("🔍 Generating explanation..."):
                    explanation = explain_text(lime_explainer, input_text, predictor, num_features=8)
                    if explanation:
                        explanation_data = format_lime_explanation(explanation, prediction_label)
            
            # Display results
            display_prediction_results(input_text, prediction_label, confidence, explanation_data)
            
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze.")

def reddit_monitoring_tab():
    """Enhanced Reddit monitoring tab."""
    st.markdown('<h2 style="text-align: center;">📱 Live Reddit Monitoring</h2>', unsafe_allow_html=True)
    
    # Load model
    model, tokenizer = load_model_cached()
    
    if model is None or tokenizer is None:
        st.error("❌ Model not loaded. Please check your model files.")
        return
    
    # Reddit API configuration
    st.markdown("""
    <div class="info-card">
        <h4>🔧 Reddit API Configuration</h4>
        <p>Monitor mental health discussions in real-time from various subreddits.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Use secrets if available
    reddit_configured = False
    if 'reddit' in st.secrets:
        client_id = st.secrets.reddit.client_id
        client_secret = st.secrets.reddit.client_secret
        user_agent = st.secrets.reddit.user_agent
        reddit_configured = True
        st.success("✅ Reddit API credentials loaded from configuration.")
    else:
        st.info("💡 Add your Reddit API credentials to secrets.toml for automatic configuration.")
        col1, col2 = st.columns(2)
        with col1:
            client_id = st.text_input("Client ID", type="password")
            user_agent = st.text_input("User Agent", value="MindTrack:v2.0")
        with col2:
            client_secret = st.text_input("Client Secret", type="password")
        
        reddit_configured = all([client_id, client_secret, user_agent])
    
    if not reddit_configured:
        st.warning("⚠️ Please provide Reddit API credentials to use monitoring features.")
        return
    
    # Subreddit configuration
    st.markdown("""
    <div class="info-card">
        <h4>📱 Subreddit Settings</h4>
        <p>Configure which subreddit to monitor and analysis parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        subreddit_options = [
            "depression", "anxiety", "mentalhealth", "SuicideWatch",
            "bipolar", "PTSD", "selfharm", "getting_over_it",
            "decidingtobebetter", "mentalillness"
        ]
        subreddit_name = st.selectbox("Select Subreddit", subreddit_options, index=2)
    
    with col2:
        max_posts = st.number_input("Max Posts to Analyze", min_value=5, max_value=20, value=10)
    
    with col3:
        analysis_mode = st.selectbox("Analysis Mode", ["New Posts", "Hot Posts", "Top Posts"])
    
    # Start analysis
    if st.button("🚀 Start Analysis", type="primary"):
        try:
            # Initialize Reddit API
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            st.success(f"✅ Connected to Reddit. Analyzing r/{subreddit_name}")
            
            subreddit = reddit.subreddit(subreddit_name)
            
            # Get posts based on mode
            if analysis_mode == "New Posts":
                posts = list(subreddit.new(limit=max_posts))
            elif analysis_mode == "Hot Posts":
                posts = list(subreddit.hot(limit=max_posts))
            else:
                posts = list(subreddit.top(time_filter='day', limit=max_posts))
            
            if not posts:
                st.warning("No posts found in the selected subreddit.")
                return
            
            # Analyze posts
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            risk_count = 0
            normal_count = 0
            
            for i, submission in enumerate(posts):
                progress_bar.progress((i + 1) / len(posts))
                status_text.text(f"Analyzing post {i + 1}/{len(posts)}")
                
                # Combine title and text
                post_text = f"{submission.title} {submission.selftext}".strip()
                
                if len(post_text) > 20:  # Only analyze substantial posts
                    prediction_label, confidence, probabilities = predict_single_text(
                        post_text, model, tokenizer
                    )
                    
                    if prediction_label == "Risk Detected":
                        risk_count += 1
                    else:
                        normal_count += 1
                    
                    # Display result
                    with results_container:
                        with st.expander(f"Post {i+1}: {submission.title[:60]}..." + 
                                       (" 🚨" if prediction_label == "Risk Detected" else " ✅")):
                            st.write(f"**Author:** u/{submission.author if submission.author else 'Unknown'}")
                            st.write(f"**Posted:** {datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M')}")
                            st.write(f"**Text:** {post_text[:200]}...")
                            
                            if prediction_label == "Risk Detected":
                                st.markdown(f'<div class="risk-high">🚨 {prediction_label} ({confidence:.1%})</div>', 
                                          unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="risk-normal">✅ {prediction_label} ({confidence:.1%})</div>', 
                                          unsafe_allow_html=True)
                            
                            st.write(f"**Reddit Link:** https://reddit.com{submission.permalink}")
            
            # Summary
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>{len(posts)}</h3>
                    <p>Posts Analyzed</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>{risk_count}</h3>
                    <p>Risk Detected</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>{normal_count}</h3>
                    <p>Normal Posts</p>
                </div>
                ''', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error connecting to Reddit: {str(e)}")

def main():
    """Main application."""
    # Header
    st.markdown('<h1 class="main-header">🧠 MindTrack</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2em; color: #666;">AI-Powered Mental Health Sentiment Analyzer</p>', 
                unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "📱 Reddit Monitor", "ℹ️ About"])
    
    with tab1:
        single_text_analysis_tab()
    
    with tab2:
        reddit_monitoring_tab()
    
    with tab3:
        st.markdown("""
        ## About MindTrack
        
        MindTrack is an AI-powered mental health sentiment analyzer that uses advanced natural language processing 
        to detect potential mental health risks in text content.
        
        ### Features:
        - **🤖 Advanced AI Model**: Fine-tuned DistilBERT for mental health sentiment analysis
        - **🔍 Explainable AI**: LIME explanations show which words influenced the prediction
        - **📱 Reddit Integration**: Monitor mental health subreddits in real-time
        - **⚡ Fast Performance**: Optimized for quick analysis and user experience
        - **🎯 High Accuracy**: 97%+ accuracy on mental health text classification
        
        ### How to Use:
        1. **Text Analysis**: Enter any text to get instant mental health sentiment analysis
        2. **Reddit Monitor**: Analyze posts from mental health subreddits in real-time
        3. **View Explanations**: Understand why the AI made its prediction
        
        ### Important Note:
        This tool is designed to assist in identifying potential mental health concerns but should not replace 
        professional medical advice or diagnosis. If you or someone you know is experiencing mental health 
        difficulties, please seek help from qualified professionals.
        
        ### Crisis Resources:
        - **National Suicide Prevention Lifeline**: 988
        - **Crisis Text Line**: Text HOME to 741741
        - **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/
        """)

if __name__ == "__main__":
    main()