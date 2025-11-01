import streamlit as st
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🤖",
    layout="wide"
)

# --- 2. LOAD CUSTOM CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css('style.css')


# --- 3. BACKEND FUNCTIONS (CACHED FOR PERFORMANCE) ---
@st.cache_data
def load_spacy_model():
    return spacy.load('en_core_web_lg')

@st.cache_data
def load_job_data(filepath):
    df = pd.read_csv(filepath)
    df['Skills'] = df['Skills'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    return df

@st.cache_resource
def create_skill_matcher(_nlp, skill_file_path):
    matcher = PhraseMatcher(_nlp.vocab, attr='LOWER')
    with open(skill_file_path, 'r') as f:
        skills = [line.strip().lower() for line in f if line.strip()]
    skill_patterns = [_nlp.make_doc(skill) for skill in skills]
    matcher.add("SKILL", skill_patterns)
    return matcher, skills

nlp = load_spacy_model()
job_df = load_job_data('jobs_with_skills.csv')
skill_matcher, all_skills = create_skill_matcher(nlp, 'skills.txt')

# --- (Other backend functions remain the same) ---
def parse_resume_pdf(uploaded_file):
    try:
        file_bytes = uploaded_file.read()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def extract_skills(text, matcher, all_skills_list):
    doc = nlp(text.lower())
    found_skills = set()
    matches = matcher(doc)
    for _, start, end in matches:
        found_skills.add(doc[start:end].text)
    for token in doc:
        if token.text.lower() in all_skills_list:
            found_skills.add(token.text.lower())
    return list(found_skills)

def get_recommendations(resume_skills, job_df, top_n=5):
    job_skills_str = job_df['Skills'].apply(lambda s: ' '.join(s))
    resume_skills_str = ' '.join(resume_skills)
    vectorizer = TfidfVectorizer(analyzer='word')
    job_vectors = vectorizer.fit_transform(job_skills_str)
    resume_vector = vectorizer.transform([resume_skills_str])
    similarity_scores = cosine_similarity(resume_vector, job_vectors)[0]
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    recommendations_df = job_df.iloc[top_indices].copy()
    recommendations_df['Match_Score'] = [round(similarity_scores[i] * 100) for i in top_indices]
    return recommendations_df

def analyze_skill_gap(resume_skills, job_skills):
    resume_set = set(resume_skills)
    job_set = set(job_skills)
    missing = list(job_set - resume_set)
    matching = list(job_set.intersection(resume_set))
    completeness = round(len(matching) / len(job_set) * 100) if job_set else 100
    return missing, matching, completeness


# --- 4. UI LAYOUT ---

# --- Header ---
st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 2rem;">
        <div style="background-color: #6D28D9; border-radius: 10px; padding: 10px; margin-right: 15px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="white" class="bi bi-briefcase-fill" viewBox="0 0 16 16">
              <path d="M6.5 1A1.5 1.5 0 0 0 5 2.5V3h6V2.5A1.5 1.5 0 0 0 9.5 1h-3z"/>
              <path d="M0 4.5A1.5 1.5 0 0 1 1.5 3h13A1.5 1.5 0 0 1 16 4.5v1.384l-7.614 2.03a1.5 1.5 0 0 1-.772 0L0 5.884V4.5zM1.5 4a.5.5 0 0 0-.5.5v1.616l6.871 1.832a.5.5 0 0 0 .258 0L15 6.116V4.5a.5.5 0 0 0-.5-.5h-13zM0 8.5v4.5A1.5 1.5 0 0 0 1.5 14h13a1.5 1.5 0 0 0 1.5-1.5V8.5L8.386 10.386a1.5 1.5 0 0 1-1.772 0L0 8.5z"/>
            </svg>
        </div>
        <div>
            <h1 style="margin-bottom: 0; font-weight: 700;">AI Career Advisor</h1>
            <p style="margin: 0; color: #9CA3AF;">Smart job matching powered by AI</p>
        </div>
    </div>
""", unsafe_allow_html=True)


# --- "Get Started" Card ---
with st.container():
    st.subheader("↗ Get Started")
    col1, col2 = st.columns(2)
    with col1:
        job_categories = [""] + sorted(job_df['Job_Category'].unique())
        selected_category = st.selectbox("Select Job Category", options=job_categories, label_visibility="collapsed")
    with col2:
        uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=['pdf'], label_visibility="collapsed")

    analyze_button = st.button("Analyze Resume", use_container_width=True)

# --- Results Section ---
if analyze_button and uploaded_resume and selected_category:
    with st.spinner('Analyzing...'):
        resume_text = parse_resume_pdf(uploaded_resume)
        if resume_text:
            resume_skills = extract_skills(resume_text, skill_matcher, all_skills)
            
            # --- "Your Skills" Card ---
            with st.container():
                st.subheader("✅ Your Skills")
                skills_html = "".join([f"<span class='skill-pill'>{skill}</span>" for skill in resume_skills])
                st.markdown(f"<div>{skills_html}</div>", unsafe_allow_html=True)

            # --- "Recommended Jobs" Card ---
            with st.container():
                st.subheader("💼 Recommended Jobs")
                filtered_jobs = job_df[job_df['Job_Category'] == selected_category].copy()
                recommendations = get_recommendations(resume_skills, filtered_jobs)

                for _, row in recommendations.iterrows():
                    with st.container():
                        st.markdown("---")
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"#### {row['Title']}")
                            st.markdown(f"**{row['Company']}**")
                            st.markdown(f"📍 {row['Location']}")
                        with col2:
                            st.progress(row['Match_Score'], text=f"{row['Match_Score']}%")
                            st.caption("Match Score")
            
            # --- "Skill Gap Analysis" Card ---
            if not recommendations.empty:
                top_job = recommendations.iloc[0]
                missing, matching, completeness = analyze_skill_gap(resume_skills, top_job['Skills'])

                with st.container():
                    st.subheader(f"💡 Skill Gap Analysis for {top_job['Title']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### Matching Skills")
                        for skill in matching:
                            st.markdown(f"""
                            <div class="skill-item skill-item-match">
                                <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M12.736 3.97a.733.733 0 0 1 1.047 0c.286.289.29.756.01 1.05L7.88 12.01a.733.733 0 0 1-1.065.02L3.217 8.384a.757.757 0 0 1 0-1.06.733.733 0 0 1 1.047 0l3.052 3.093 5.4-6.425z"/></svg>
                                {skill}
                            </div>
                            """, unsafe_allow_html=True)
                    with col2:
                        st.markdown("##### Missing Skills")
                        for skill in missing:
                            st.markdown(f"""
                            <div class="skill-item skill-item-missing">
                               <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16zM5.255 5.786a.237.237 0 0 0 .241.246h.825a.237.237 0 0 0 .241-.246V5.536a.237.237 0 0 0-.241-.246h-.825a.237.237 0 0 0-.241.246v.25zM8 4a.905.905 0 0 0-.9.995l.35 3.507a.552.552 0 0 0 1.1 0l.35-3.507A.905.905 0 0 0 8 4zm.002 6.03a.902.902 0 1 0 0 1.804.902.902 0 0 0 0-1.804z"/></svg>
                                {skill}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("##### Overall Skill Completeness")
                    st.progress(completeness, text=f"{completeness}%")

elif analyze_button:
    st.warning("Please select a job category and upload your resume.")