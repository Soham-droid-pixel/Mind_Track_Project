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

# Enhanced Custom CSS for Dark Mode
st.markdown("""
<style>
/* Dark Mode Main App Styling */
.stApp {
    background: linear-gradient(135deg, #0f1419 0%, #1a1a2e 50%, #16213e 100%);
    background-attachment: fixed;
    color: #e0e6ed;
}

.main {
    background: rgba(20, 25, 35, 0.95);
    border-radius: 20px;
    margin: 20px;
    padding: 30px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Dark mode text colors */
h1, h2, h3, h4, h5, h6, p, span, div {
    color: #e0e6ed !important;
}

/* Streamlit elements dark mode */
.stMarkdown {
    color: #e0e6ed;
}

.stTextArea textarea, .stTextInput input, .stSelectbox select {
    background-color: #2d3748 !important;
    color: #e0e6ed !important;
    border: 2px solid #4a5568 !important;
}

.main-header {
    font-size: 3.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 800;
    text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    position: relative;
}

.main-header::before {
    content: "🧠";
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    font-size: 3.5rem;
    z-index: -1;
    opacity: 0.2;
}

.subtitle {
    text-align: center;
    font-size: 1.3rem;
    color: #a0aec0;
    margin-bottom: 2rem;
    font-weight: 300;
}

.risk-high {
    color: #fc8181;
    font-weight: bold;
    font-size: 1.4em;
    background: linear-gradient(135deg, #2d1b69 0%, #11101d 100%);
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #fc8181;
    box-shadow: 0 8px 25px rgba(252, 129, 129, 0.3);
    text-align: center;
    margin: 15px 0;
    animation: pulse-dark 2s infinite;
}

.risk-normal {
    color: #68d391;
    font-weight: bold;
    font-size: 1.4em;
    background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #68d391;
    box-shadow: 0 8px 25px rgba(104, 211, 145, 0.3);
    text-align: center;
    margin: 15px 0;
}

@keyframes pulse-dark {
    0% { box-shadow: 0 8px 25px rgba(252, 129, 129, 0.3); }
    50% { box-shadow: 0 8px 35px rgba(252, 129, 129, 0.6); }
    100% { box-shadow: 0 8px 25px rgba(252, 129, 129, 0.3); }
}

.confidence-score {
    font-size: 1.3em;
    font-weight: 600;
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    background: #2d3748;
    color: #e0e6ed;
}

.confidence-high {
    border: 2px solid #68d391;
    box-shadow: 0 8px 25px rgba(104, 211, 145, 0.3);
}

.confidence-medium {
    border: 2px solid #f6ad55;
    box-shadow: 0 8px 25px rgba(246, 173, 85, 0.3);
}

.confidence-low {
    border: 2px solid #fc8181;
    box-shadow: 0 8px 25px rgba(252, 129, 129, 0.3);
}

.explanation-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2px;
    border-radius: 20px;
    margin: 20px 0;
    box-shadow: 0 15px 35px rgba(0,0,0,0.4);
}

.explanation-content {
    background: #1a202c;
    padding: 25px;
    border-radius: 18px;
    margin: 0;
    color: #e0e6ed;
}

.word-tag {
    display: inline-block;
    padding: 8px 15px;
    margin: 5px;
    border-radius: 25px;
    font-size: 0.95em;
    font-weight: 600;
    transition: all 0.3s ease;
    cursor: pointer;
}

.word-tag:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

.positive-influence {
    background: linear-gradient(135deg, #2d1b69 0%, #11101d 100%);
    color: #fc8181;
    border: 1px solid #fc8181;
}

.negative-influence {
    background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
    color: #68d391;
    border: 1px solid #68d391;
}

.quick-examples {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 25px;
    border-radius: 20px;
    margin: 25px 0;
    color: white;
    box-shadow: 0 15px 35px rgba(0,0,0,0.4);
}

.quick-examples h4 {
    color: white !important;
    margin-bottom: 15px;
    font-size: 1.3em;
}

.quick-examples p {
    color: rgba(255,255,255,0.9) !important;
}
.example-button {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 25px;
    padding: 12px 20px;
    margin: 8px;
    cursor: pointer;
    font-size: 0.95em;
    color: white;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.example-button:hover {
    background: rgba(255,255,255,0.25);
    transform: translateY(-2px);
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin: 15px 0;
    box-shadow: 0 15px 35px rgba(0,0,0,0.4);
    transition: all 0.3s ease;
    border: 1px solid rgba(255,255,255,0.1);
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.6);
}

.metric-card h3 {
    font-size: 2.5em;
    margin: 0;
    font-weight: 700;
    color: white !important;
}

.metric-card p {
    color: rgba(255,255,255,0.9) !important;
}

.info-card {
    background: rgba(45, 55, 72, 0.9);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
    margin: 20px 0;
    backdrop-filter: blur(10px);
}

.info-card h3, .info-card h4 {
    color: #667eea !important;
}

.info-card p {
    color: #a0aec0 !important;
}

/* Enhanced Dark Mode button styling */
.stButton > button {
    width: 100%;
    border-radius: 15px;
    height: 3.5em;
    font-weight: 600;
    font-size: 1.1em;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
}

/* Dark Mode Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(45, 55, 72, 0.8);
    color: #e0e6ed !important;
    border-radius: 15px;
    padding: 15px 25px;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.1);
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.6);
}

/* Dark Mode Text area styling */
.stTextArea > div > div > textarea {
    border-radius: 15px;
    background-color: #2d3748 !important;
    color: #e0e6ed !important;
    border: 2px solid #4a5568 !important;
    font-size: 1.1em;
    padding: 15px;
    transition: all 0.3s ease;
}

.stTextArea > div > div > textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.4) !important;
}

/* Dark Mode Select box styling */
.stSelectbox > div > div {
    border-radius: 15px;
    background-color: #2d3748 !important;
    border: 2px solid #4a5568 !important;
}

.stSelectbox > div > div > div {
    background-color: #2d3748 !important;
    color: #e0e6ed !important;
}

/* Dark Mode number input */
.stNumberInput > div > div > input {
    background-color: #2d3748 !important;
    color: #e0e6ed !important;
    border: 2px solid #4a5568 !important;
    border-radius: 10px;
}

/* Dark Mode Success/Error messages */
.stSuccess {
    background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
    border: 1px solid #68d391;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(104, 211, 145, 0.2);
    color: #68d391 !important;
}

.stError {
    background: linear-gradient(135deg, #2d1b69 0%, #11101d 100%);
    border: 1px solid #fc8181;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(252, 129, 129, 0.2);
    color: #fc8181 !important;
}

.stWarning {
    background: linear-gradient(135deg, #2d2a1b 0%, #1d1b11 100%);
    border: 1px solid #f6ad55;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(246, 173, 85, 0.2);
    color: #f6ad55 !important;
}

.stInfo {
    background: linear-gradient(135deg, #1a2332 0%, #2d3748 100%);
    border: 1px solid #63b3ed;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(99, 179, 237, 0.2);
    color: #63b3ed !important;
}

/* Dark Mode Expander styling */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    color: white !important;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.1);
}

.streamlit-expanderContent {
    background: rgba(45, 55, 72, 0.9);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 0 0 15px 15px;
    color: #e0e6ed;
}

/* Progress bar dark mode */
.stProgress .st-bo {
    background-color: #2d3748;
}

.stProgress .st-bp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Sidebar dark mode */
.css-1d391kg {
    background: rgba(20, 25, 35, 0.95);
}

/* Fix for white text elements */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #e0e6ed !important;
}

.stMarkdown p {
    color: #a0aec0 !important;
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
        st.markdown('<div class="explanation-content">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

def single_text_analysis_tab():
    """Enhanced single text analysis tab."""
    st.markdown('''
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="font-size: 2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;">
            📝 Single Text Analysis
        </h2>
        <p style="font-size: 1.2rem; color: #6c757d; margin: 0;">
            Analyze any text for mental health sentiment indicators
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model, tokenizer = load_model_cached()
    
    if model is None or tokenizer is None:
        st.error("❌ Model not loaded. Please check your model files.")
        return
    
    # Quick examples with improved styling
    st.markdown("""
    <div class="quick-examples">
        <h4>💡 Quick Examples</h4>
        <p style="margin-bottom: 20px; opacity: 0.9;">Click any example below to test it instantly!</p>
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
    # Enhanced Header with subtitle and visible logo
    st.markdown('''
    <div style="text-align: center; margin-bottom: 40px;">
        <div style="font-size: 4rem; margin-bottom: 10px;">🧠</div>
        <h1 class="main-header" style="margin-top: 0;">MindTrack</h1>
        <p class="subtitle">AI-Powered Mental Health Sentiment Analyzer</p>
        <div style="width: 100px; height: 4px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: 20px auto; border-radius: 2px;"></div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Enhanced Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "📱 Reddit Monitor", "ℹ️ About"])
    
    with tab1:
        single_text_analysis_tab()
    
    with tab2:
        reddit_monitoring_tab()
    
    with tab3:
        st.markdown('''
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 style="font-size: 2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;">
                ℹ️ About MindTrack
            </h2>
            <p style="font-size: 1.2rem; color: #a0aec0; margin: 0;">
                Advanced AI-powered mental health sentiment analysis
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Key Features
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #667eea; margin-bottom: 20px;">🚀 Key Features</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                <div style="padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white;">
                    <h4 style="color: white !important;">🤖 Advanced AI Model</h4>
                    <p style="margin: 0; opacity: 0.9; color: rgba(255,255,255,0.9) !important;">Fine-tuned DistilBERT for mental health sentiment analysis</p>
                </div>
                <div style="padding: 15px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 12px; color: white;">
                    <h4 style="color: white !important;">🔍 Explainable AI</h4>
                    <p style="margin: 0; opacity: 0.9; color: rgba(255,255,255,0.9) !important;">LIME explanations show which words influenced predictions</p>
                </div>
                <div style="padding: 15px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 12px; color: white;">
                    <h4 style="color: white !important;">📱 Reddit Integration</h4>
                    <p style="margin: 0; opacity: 0.9; color: rgba(255,255,255,0.9) !important;">Monitor mental health subreddits in real-time</p>
                </div>
                <div style="padding: 15px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); border-radius: 12px; color: white;">
                    <h4 style="color: white !important;">⚡ High Accuracy</h4>
                    <p style="margin: 0; opacity: 0.9; color: rgba(255,255,255,0.9) !important;">97%+ accuracy on mental health text classification</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # How to Use
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #667eea; margin-bottom: 20px;">📖 How to Use</h3>
            <div style="display: flex; flex-direction: column; gap: 15px;">
                <div style="display: flex; align-items: center; padding: 15px; background: rgba(102, 126, 234, 0.2); border-radius: 12px; border-left: 4px solid #667eea;">
                    <span style="font-size: 2em; margin-right: 15px;">📝</span>
                    <div>
                        <h4 style="margin: 0; color: #e0e6ed !important;">Text Analysis</h4>
                        <p style="margin: 5px 0 0 0; color: #a0aec0 !important;">Enter any text to get instant mental health sentiment analysis</p>
                    </div>
                </div>
                <div style="display: flex; align-items: center; padding: 15px; background: rgba(245, 87, 108, 0.2); border-radius: 12px; border-left: 4px solid #f5576c;">
                    <span style="font-size: 2em; margin-right: 15px;">📱</span>
                    <div>
                        <h4 style="margin: 0; color: #e0e6ed !important;">Reddit Monitor</h4>
                        <p style="margin: 5px 0 0 0; color: #a0aec0 !important;">Analyze posts from mental health subreddits in real-time</p>
                    </div>
                </div>
                <div style="display: flex; align-items: center; padding: 15px; background: rgba(79, 172, 254, 0.2); border-radius: 12px; border-left: 4px solid #4facfe;">
                    <span style="font-size: 2em; margin-right: 15px;">🔍</span>
                    <div>
                        <h4 style="margin: 0; color: #e0e6ed !important;">View Explanations</h4>
                        <p style="margin: 5px 0 0 0; color: #a0aec0 !important;">Understand why the AI made its prediction with detailed analysis</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Important Disclaimer
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2d1b69 0%, #11101d 100%); padding: 25px; border-radius: 20px; margin: 25px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.4); border: 2px solid #fc8181;">
            <div style="text-align: center; margin-bottom: 20px;">
                <h3 style="color: #fc8181 !important; margin: 0;">⚠️ Important Disclaimer</h3>
            </div>
            <p style="color: #e0e6ed !important; font-size: 1.1em; line-height: 1.6; margin: 0; text-align: center;">
                This tool is designed to <strong style="color: #fc8181;">assist</strong> in identifying potential mental health concerns but should 
                <strong style="color: #fc8181;">not replace professional medical advice or diagnosis</strong>. If you or someone you know is 
                experiencing mental health difficulties, please seek help from qualified professionals.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Crisis Resources
        st.markdown("""
        <div class="info-card">
            <h3 style="color: #fc8181; margin-bottom: 20px;">🆘 Crisis Resources</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #2d1b69 0%, #11101d 100%); border-radius: 15px; border: 2px solid #fc8181;">
                    <h4 style="margin: 0 0 10px 0; color: #e0e6ed !important;">📞 US Crisis Hotline</h4>
                    <p style="margin: 0; font-size: 1.3em; font-weight: bold; color: #fc8181 !important;">988</p>
                </div>
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); border-radius: 15px; border: 2px solid #68d391;">
                    <h4 style="margin: 0 0 10px 0; color: #e0e6ed !important;">💬 Crisis Text Line</h4>
                    <p style="margin: 0; font-size: 1.1em; font-weight: bold; color: #68d391 !important;">Text HOME to 741741</p>
                </div>
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #2d2a1b 0%, #1d1b11 100%); border-radius: 15px; border: 2px solid #f6ad55;">
                    <h4 style="margin: 0 0 10px 0; color: #e0e6ed !important;">🌍 International</h4>
                    <p style="margin: 0; font-size: 0.95em; color: #f6ad55 !important;">IASP Crisis Centers</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()