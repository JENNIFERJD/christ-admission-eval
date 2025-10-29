"""
Christ University - AI Admission Evaluation System
FREE system using Google Gemini API
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
        st.warning("⚠️ Transcript too short")
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

Respond with ONLY this JSON:
{{
  "communication_skills": {{"score": 7.5, "justification": "explanation"}},
  "subject_knowledge": {{"score": 8.0, "justification": "explanation"}},
  "confidence": {{"score": 7.0, "justification": "explanation"}},
  "clarity_of_thought": {{"score": 8.5, "justification": "explanation"}},
  "strengths": ["strength1", "strength2", "strength3"],
  "improvements": ["improvement1", "improvement2"],
  "recommendation": "Strongly Recommend/Recommend/Consider with Reservations/Not Recommended"
}}"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        evaluation = json.loads(clean_json(response.text))
        weights = {'communication_skills': 0.25, 'subject_knowledge': 0.35, 'confidence': 0.15, 'clarity_of_thought': 0.25}
        evaluation['overall_score'] = round(sum(evaluation[k]['score'] * weights[k] for k in weights.keys()), 2)
        evaluation['features'] = features
        return evaluation
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def evaluate_sop(sop_text, program="General"):
    """Evaluate SOP"""
    if not GEMINI_API_KEY:
        st.error("⚠️ API Key not configured")
        return None
    if len(sop_text.strip()) < 100:
        st.warning("⚠️ SOP too short")
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

Respond with ONLY this JSON:
{{
  "content_quality": {{"score": 8.0, "justification": "explanation"}},
  "writing_quality": {{"score": 7.5, "justification": "explanation"}},
  "originality": {{"score": 8.5, "justification": "explanation"}},
  "structure": {{"score": 8.0, "justification": "explanation"}},
  "strengths": ["strength1", "strength2", "strength3"],
  "improvements": ["improvement1", "improvement2"],
  "flags": [],
  "recommendation": "Strongly Recommend/Recommend/Consider with Reservations/Not Recommended"
}}"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        evaluation = json.loads(clean_json(response.text))
        weights = {'content_quality': 0.35, 'writing_quality': 0.25, 'originality': 0.25, 'structure': 0.15}
        evaluation['overall_score'] = round(sum(evaluation[k]['score'] * weights[k] for k in weights.keys()), 2)
        evaluation['features'] = features
        return evaluation
    except Exception as e:
        st.error(f"Error: {str(e)}")
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
        2. In Streamlit Cloud: Settings → Secrets → Add:
           ```
           GEMINI_API_KEY = "your-key-here"
           ```
        3. Click "Save" and app will restart
        """)
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📹 Interview Evaluation", "📝 SOP Evaluation", "ℹ️ Help"])
    
    # INTERVIEW TAB
    with tab1:
        st.header("Interview Evaluation")
        transcript = st.text_area("Interview Transcript", height=300, 
            placeholder="Paste the interview transcript here...")
        questions = st.text_area("Interview Questions (Optional)", height=100)
        
        if transcript:
            f = extract_features(transcript)
            col1, col2, col3 = st.columns(3)
            col1.metric("Words", f['word_count'])
            col2.metric("Fillers", f['filler_count'])
            col3.metric("Diversity", f"{f['lexical_diversity']}%")
        
        if st.button("🚀 Evaluate Interview", type="primary", disabled=not transcript):
            with st.spinner("Evaluating with Gemini AI..."):
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
                        st.metric("Communication (25%)", f"{result['communication_skills']['score']}/10")
                        st.caption(result['communication_skills']['justification'])
                        st.metric("Subject Knowledge (35%)", f"{result['subject_knowledge']['score']}/10")
                        st.caption(result['subject_knowledge']['justification'])
                    with col2:
                        st.metric("Confidence (15%)", f"{result['confidence']['score']}/10")
                        st.caption(result['confidence']['justification'])
                        st.metric("Clarity (25%)", f"{result['clarity_of_thought']['score']}/10")
                        st.caption(result['clarity_of_thought']['justification'])
                    
                    st.divider()
                    
                    # Feedback
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success("**💪 Strengths**")
                        for s in result['strengths']:
                            st.markdown(f"• {s}")
                    with col2:
                        st.warning("**🎯 Improvements**")
                        for i in result['improvements']:
                            st.markdown(f"• {i}")
                    
                    # Download
                    st.download_button(
                        "📥 Download Report",
                        json.dumps(result, indent=2),
                        f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
    
    # SOP TAB
    with tab2:
        st.header("SOP Evaluation")
        sop = st.text_area("Statement of Purpose", height=400,
            placeholder="Paste the SOP here...")
        program = st.selectbox("Program", 
            ["Computer Science", "Business", "Data Science", "Engineering", "Other"])
        
        if sop:
            f = extract_features(sop)
            col1, col2 = st.columns(2)
            col1.metric("Words", f['word_count'])
            status = "✅ Good" if 500 <= f['word_count'] <= 800 else "⚠️ Check"
            col2.metric("Length", status)
        
        if st.button("🚀 Evaluate SOP", type="primary", disabled=not sop):
            with st.spinner("Evaluating with Gemini AI..."):
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
                        st.metric("Content (35%)", f"{result['content_quality']['score']}/10")
                        st.caption(result['content_quality']['justification'])
                        st.metric("Writing (25%)", f"{result['writing_quality']['score']}/10")
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
                        st.success("**💪 Strengths**")
                        for s in result['strengths']:
                            st.markdown(f"• {s}")
                    with col2:
                        st.warning("**🎯 Improvements**")
                        for i in result['improvements']:
                            st.markdown(f"• {i}")
                    
                    if result.get('flags'):
                        st.error("**🚩 Flags**")
                        for flag in result['flags']:
                            st.markdown(f"• {flag}")
                    
                    # Download
                    st.download_button(
                        "📥 Download Report",
                        json.dumps(result, indent=2),
                        f"sop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
    
    # HELP TAB
    with tab3:
        st.header("About This System")
        st.markdown("""
        ### 🆓 Completely FREE
        - **Google Gemini**: FREE API (1,500 requests/day)
        - **Streamlit Cloud**: FREE hosting
        - **No credit card required**
        
        ### 📊 How It Works
        1. Extract objective metrics (words, fillers, diversity)
        2. AI analyzes content quality and structure
        3. Generate weighted scores across criteria
        4. Provide actionable feedback
        
        ### 🎯 Evaluation Criteria
        
        **Interview (0-10 scale):**
        - Communication Skills (25%)
        - Subject Knowledge (35%)
        - Confidence (15%)
        - Clarity of Thought (25%)
        
        **SOP (0-10 scale):**
        - Content Quality (35%)
        - Writing Quality (25%)
        - Originality (25%)
        - Structure (15%)
        
        ### 🚀 Free Tier Limits
        - 1,500 evaluations per day
        - 15 requests per minute
        - Unlimited forever!
        
        ### 📧 Support
        Built for Christ University Admission System
        """)
        
        st.info("💡 **Tip**: For transcription, use FREE Groq Whisper API!")

if __name__ == "__main__":
    main()
