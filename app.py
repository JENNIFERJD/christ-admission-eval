"""
Christ University - AI Admission Evaluation System
FREE system using Google Gemini API
FIXED: Updated to use correct model name
"""

import streamlit as st
import json
import re
from datetime import datetime

# Check if google.generativeai is available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ==================== CONFIGURATION ====================

st.set_page_config(
    page_title="Christ University - AI Evaluation",
    page_icon="🎓",
    layout="wide"
)

# Get API key
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    if GEMINI_AVAILABLE:
        genai.configure(api_key=GEMINI_API_KEY)
except:
    GEMINI_API_KEY = None

# ==================== HELPER FUNCTIONS ====================

def extract_features(text):
    """Extract objective metrics"""
    if not text:
        return None
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    fillers = ['um', 'uh', 'like', 'you know', 'basically', 'actually']
    filler_count = sum(len(re.findall(r'\b' + f + r'\b', text.lower())) for f in fillers)
    unique = set(w.lower() for w in words if w.isalpha())
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_sentence_length': round(len(words) / max(len(sentences), 1), 1),
        'filler_count': filler_count,
        'unique_words': len(unique),
        'lexical_diversity': round(len(unique) / max(len(words), 1) * 100, 1)
    }

def clean_json(text):
    """Clean JSON from response"""
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    start = text.find('{')
    end = text.rfind('}') + 1
    if start != -1 and end > start:
        return text[start:end]
    return text

def evaluate_interview(transcript, questions=""):
    """Evaluate interview"""
    if not GEMINI_API_KEY:
        st.error("⚠️ API Key not configured")
        return None
    if len(transcript.strip()) < 50:
        st.warning("⚠️ Transcript too short (minimum 50 characters)")
        return None
    
    features = extract_features(transcript)
    prompt = f"""You are an expert admission evaluator for Christ University.

INTERVIEW TRANSCRIPT:
{transcript}

QUESTIONS: {questions if questions else "General interview"}

METRICS: Words={features['word_count']}, Fillers={features['filler_count']}, Diversity={features['lexical_diversity']}%

Rate on 4 criteria (0-10 each):
1. COMMUNICATION SKILLS (25%): Grammar, vocabulary, fluency
2. SUBJECT KNOWLEDGE (35%): Depth, accuracy, examples
3. CONFIDENCE (15%): Organization, minimal hesitation
4. CLARITY OF THOUGHT (25%): Logic, relevance, critical thinking

Respond with ONLY this JSON (no extra text):
{{
  "communication_skills": {{"score": 7.5, "justification": "explanation"}},
  "subject_knowledge": {{"score": 8.0, "justification": "explanation"}},
  "confidence": {{"score": 7.0, "justification": "explanation"}},
  "clarity_of_thought": {{"score": 8.5, "justification": "explanation"}},
  "strengths": ["strength1", "strength2", "strength3"],
  "improvements": ["improvement1", "improvement2"],
  "recommendation": "Strongly Recommend OR Recommend OR Consider with Reservations OR Not Recommended"
}}"""
    
    try:
        # FIXED: Using correct model name
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        evaluation = json.loads(clean_json(response.text))
        weights = {'communication_skills': 0.25, 'subject_knowledge': 0.35, 'confidence': 0.15, 'clarity_of_thought': 0.25}
        evaluation['overall_score'] = round(sum(evaluation[k]['score'] * weights[k] for k in weights.keys()), 2)
        evaluation['features'] = features
        return evaluation
    except json.JSONDecodeError as e:
        st.error(f"❌ Could not parse AI response. Please try again.")
        with st.expander("Debug Info"):
            st.code(response.text if 'response' in locals() else "No response")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def evaluate_sop(sop_text, program="General"):
    """Evaluate SOP"""
    if not GEMINI_API_KEY:
        st.error("⚠️ API Key not configured")
        return None
    if len(sop_text.strip()) < 100:
        st.warning("⚠️ SOP too short (minimum 100 characters)")
        return None
    
    features = extract_features(sop_text)
    prompt = f"""Evaluate this Statement of Purpose for Christ University's {program} program.

SOP:
{sop_text}

METRICS: Words={features['word_count']}, Sentences={features['sentence_count']}

Rate on 4 criteria (0-10 each):
1. CONTENT QUALITY (35%): Goals, experiences, program fit, motivation
2. WRITING QUALITY (25%): Grammar, vocabulary, tone, readability
3. ORIGINALITY (25%): Personal examples, unique voice, authenticity
4. STRUCTURE (15%): Organization, transitions, flow

Respond with ONLY this JSON (no extra text):
{{
  "content_quality": {{"score": 8.0, "justification": "explanation"}},
  "writing_quality": {{"score": 7.5, "justification": "explanation"}},
  "originality": {{"score": 8.5, "justification": "explanation"}},
  "structure": {{"score": 8.0, "justification": "explanation"}},
  "strengths": ["strength1", "strength2", "strength3"],
  "improvements": ["improvement1", "improvement2"],
  "flags": [],
  "recommendation": "Strongly Recommend OR Recommend OR Consider with Reservations OR Not Recommended"
}}"""
    
    try:
        # FIXED: Using correct model name
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        evaluation = json.loads(clean_json(response.text))
        weights = {'content_quality': 0.35, 'writing_quality': 0.25, 'originality': 0.25, 'structure': 0.15}
        evaluation['overall_score'] = round(sum(evaluation[k]['score'] * weights[k] for k in weights.keys()), 2)
        evaluation['features'] = features
        return evaluation
    except json.JSONDecodeError as e:
        st.error(f"❌ Could not parse AI response. Please try again.")
        with st.expander("Debug Info"):
            st.code(response.text if 'response' in locals() else "No response")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# ==================== MAIN APP ====================

def main():
    # Header
    st.title("🎓 Christ University - AI Admission Evaluation")
    st.markdown("**Intelligent Interview & SOP Assessment System**")
    
    # API Key check
    if not GEMINI_API_KEY:
        st.error("⚠️ **API Key Not Configured!**")
        st.info("""
        **Setup Instructions:**
        1. Get FREE API key: https://aistudio.google.com/app/apikey
        2. In Streamlit Cloud: Menu (⋮) → Settings → Secrets → Add:
           ```
           GEMINI_API_KEY = "your-key-here"
           ```
        3. Click "Save" and app will restart
        """)
        st.stop()
    
    if not GEMINI_AVAILABLE:
        st.error("⚠️ google-generativeai not installed. Check requirements.txt")
        st.stop()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📹 Interview Evaluation", "📝 SOP Evaluation", "ℹ️ Help"])
    
    # INTERVIEW TAB
    with tab1:
        st.header("Interview Evaluation")
        
        # Sample button
        if st.button("📝 Load Sample Interview"):
            st.session_state.sample_transcript = """Well, I've always been fascinated by computer science. During my undergraduate studies at XYZ University, I worked on a machine learning project where we developed a sentiment analysis model for social media data. The project taught me the importance of data preprocessing and feature engineering. 

I believe Christ University's program will help me develop my skills further in artificial intelligence and machine learning, which aligns with my career goal of becoming a data scientist. I'm particularly interested in the research opportunities available at your institution, especially in the areas of natural language processing and deep learning.

During my internship at TechCorp, I gained practical experience in developing predictive models and working with large datasets. This experience reinforced my passion for applying AI to solve real-world problems."""
        
        transcript = st.text_area(
            "Interview Transcript", 
            height=300,
            value=st.session_state.get('sample_transcript', ''),
            placeholder="Paste the interview transcript here...\n\nTip: Click 'Load Sample Interview' to see an example"
        )
        
        questions = st.text_area(
            "Interview Questions (Optional)", 
            height=100,
            placeholder="e.g., Why are you interested in this program? Describe a relevant project."
        )
        
        if transcript:
            f = extract_features(transcript)
            col1, col2, col3 = st.columns(3)
            col1.metric("Words", f['word_count'])
            col2.metric("Filler Words", f['filler_count'])
            col3.metric("Diversity", f"{f['lexical_diversity']}%")
        
        if st.button("🚀 Evaluate Interview", type="primary", disabled=not transcript):
            with st.spinner("Evaluating with Gemini AI... (5-10 seconds)"):
                result = evaluate_interview(transcript, questions)
                if result:
                    st.success("✅ Evaluation Complete!")
                    
                    # Overall
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Score", f"{result['overall_score']}/10")
                    with col2:
                        st.metric("Recommendation", result['recommendation'])
                    with col3:
                        score = result['overall_score']
                        cat = "🟢 Excellent" if score >= 8.5 else "🔵 Good" if score >= 7 else "🟡 Average" if score >= 6 else "🔴 Below"
                        st.metric("Category", cat)
                    
                    st.divider()
                    
                    # Scores
                    st.subheader("📊 Detailed Scores")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Communication Skills (25%)", f"{result['communication_skills']['score']}/10")
                        st.caption(result['communication_skills']['justification'])
                        st.metric("Subject Knowledge (35%)", f"{result['subject_knowledge']['score']}/10")
                        st.caption(result['subject_knowledge']['justification'])
                    with col2:
                        st.metric("Confidence (15%)", f"{result['confidence']['score']}/10")
                        st.caption(result['confidence']['justification'])
                        st.metric("Clarity of Thought (25%)", f"{result['clarity_of_thought']['score']}/10")
                        st.caption(result['clarity_of_thought']['justification'])
                    
                    st.divider()
                    
                    # Feedback
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success("**💪 Key Strengths**")
                        for s in result['strengths']:
                            st.markdown(f"• {s}")
                    with col2:
                        st.warning("**🎯 Areas for Improvement**")
                        for i in result['improvements']:
                            st.markdown(f"• {i}")
                    
                    # Download
                    st.download_button(
                        "📥 Download Report (JSON)",
                        json.dumps(result, indent=2),
                        f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    # SOP TAB
    with tab2:
        st.header("SOP Evaluation")
        
        # Sample button
        if st.button("📝 Load Sample SOP"):
            st.session_state.sample_sop = """My fascination with artificial intelligence began during a summer internship at TechCorp, where I witnessed firsthand how machine learning algorithms transformed customer service operations. This experience ignited my passion for developing intelligent systems that can solve real-world problems.

Throughout my undergraduate studies in Computer Science at XYZ University, I maintained a strong academic record with a GPA of 3.8/4.0 while actively participating in research projects. My final year project on natural language processing received recognition at the national symposium and was published in the International Journal of AI Research.

During my internship, I worked on developing a chatbot using deep learning techniques that improved customer satisfaction scores by 35%. This practical experience taught me the importance of combining theoretical knowledge with real-world applications, and reinforced my desire to pursue advanced studies in this field.

I am particularly drawn to Christ University's Data Science program because of its emphasis on practical applications and industry collaboration. The program's curriculum, especially courses in Advanced Machine Learning and Big Data Analytics, aligns perfectly with my goal of becoming a research scientist in AI. I am excited about the opportunity to work with Professor Smith's research group on natural language understanding.

My long-term goal is to contribute to the development of AI systems that can understand and process human language more naturally. I believe Christ University's rigorous academic environment and research facilities will provide the perfect foundation for achieving these aspirations."""
        
        sop = st.text_area(
            "Statement of Purpose", 
            height=400,
            value=st.session_state.get('sample_sop', ''),
            placeholder="Paste the SOP here...\n\nTip: Click 'Load Sample SOP' to see an example"
        )
        
        program = st.selectbox(
            "Program", 
            ["Computer Science", "Business Administration", "Data Science", "Engineering", "Arts & Humanities", "Other"]
        )
        
        if sop:
            f = extract_features(sop)
            col1, col2, col3 = st.columns(3)
            col1.metric("Words", f['word_count'])
            status = "✅ Good Length" if 500 <= f['word_count'] <= 800 else "⚠️ Check Length"
            col2.metric("Length Status", status)
            col3.metric("Sentences", f['sentence_count'])
        
        if st.button("🚀 Evaluate SOP", type="primary", disabled=not sop):
            with st.spinner("Evaluating with Gemini AI... (5-10 seconds)"):
                result = evaluate_sop(sop, program)
                if result:
                    st.success("✅ Evaluation Complete!")
                    
                    # Overall
                    col1, col2 = st.columns(2)
                    col1.metric("Overall Score", f"{result['overall_score']}/10")
                    col2.metric("Recommendation", result['recommendation'])
                    
                    st.divider()
                    
                    # Scores
                    st.subheader("📊 Detailed Scores")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Content Quality (35%)", f"{result['content_quality']['score']}/10")
                        st.caption(result['content_quality']['justification'])
                        st.metric("Writing Quality (25%)", f"{result['writing_quality']['score']}/10")
                        st.caption(result['writing_quality']['justification'])
                    with col2:
                        st.metric("Originality (25%)", f"{result['originality']['score']}/10")
                        st.caption(result['originality']['justification'])
                        st.metric("Structure (15%)", f"{result['structure']['score']}/10")
                        st.caption(result['structure']['justification'])
                    
                    st.divider()
                    
                    # Feedback
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success("**💪 Key Strengths**")
                        for s in result['strengths']:
                            st.markdown(f"• {s}")
                    with col2:
                        st.warning("**🎯 Areas for Improvement**")
                        for i in result['improvements']:
                            st.markdown(f"• {i}")
                    
                    if result.get('flags') and len(result['flags']) > 0:
                        st.error("**🚩 Concerns/Flags**")
                        for flag in result['flags']:
                            st.markdown(f"• {flag}")
                    
                    # Download
                    st.download_button(
                        "📥 Download Report (JSON)",
                        json.dumps(result, indent=2),
                        f"sop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    # HELP TAB
    with tab3:
        st.header("ℹ️ About This System")
        st.markdown("""
        ### 🆓 Completely FREE Solution
        - **Google Gemini Pro**: FREE API with generous limits
        - **Streamlit Cloud**: FREE hosting forever
        - **No credit card required**
        
        ### 📊 How It Works
        1. **Extract Metrics**: Word count, filler words, lexical diversity
        2. **AI Analysis**: Gemini Pro analyzes quality and structure
        3. **Weighted Scoring**: Different criteria have different weights
        4. **Actionable Feedback**: Specific strengths and improvements
        
        ### 🎯 Evaluation Criteria
        
        **Interview Assessment:**
        - Communication Skills (25%) - Grammar, vocabulary, fluency
        - Subject Knowledge (35%) - Depth, accuracy, examples
        - Confidence (15%) - Organization, minimal hesitation
        - Clarity of Thought (25%) - Logic, relevance, critical thinking
        
        **SOP Assessment:**
        - Content Quality (35%) - Goals, experiences, program fit
        - Writing Quality (25%) - Grammar, vocabulary, tone
        - Originality (25%) - Personal examples, unique voice
        - Structure (15%) - Organization, transitions, flow
        
        ### 🚀 Free Tier Limits
        - **Gemini Pro**: 60 requests per minute
        - **Daily**: Up to 1,500 requests
        - **Cost**: $0 forever
        
        ### 💡 Tips for Best Results
        - Provide detailed transcripts (200+ words)
        - Include interview questions for context
        - SOPs should be 500-800 words
        - Try the sample data to see how it works
        
        ### 🔒 Privacy & Security
        - Data sent to Google AI for processing
        - No data stored by Google after processing
        - Results stay in your browser
        - Download reports for your records
        
        ### 📧 Technical Details
        - **Model**: Google Gemini Pro
        - **Framework**: Streamlit
        - **Hosting**: Streamlit Cloud
        - **Code**: Open source on GitHub
        
        ### 🆘 Troubleshooting
        - **Slow response?** First request takes 5-10 seconds
        - **Error parsing?** Click evaluate again
        - **API error?** Check your API key in secrets
        
        ---
        
        Built for **Christ University Admission System**  
        Version 1.0 | FREE & Open Source
        """)
        
        st.info("💡 **Pro Tip**: For video transcription, use FREE Groq Whisper API or AssemblyAI free tier!")

if __name__ == "__main__":
    main()
