import streamlit as st
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Use Streamlit's cache to load models and data only once ---
@st.cache_data
def load_spacy_model():
    print("Loading spaCy model...")
    model = spacy.load('en_core_web_lg')
    print("Model loaded.")
    return model

@st.cache_data
def load_job_data(filepath):
    print(f"Loading job data from {filepath}...")
    df = pd.read_csv(filepath)
    df['Skills'] = df['Skills'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    print("Job data loaded.")
    return df

nlp = load_spacy_model()
job_df = load_job_data('jobs_with_skills.csv')


# --- All our backend functions from recommender.py go here ---
# (No changes needed for these functions)

def create_skill_matcher(skill_file_path):
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    with open(skill_file_path, 'r') as f:
        skills = [line.strip().lower() for line in f if line.strip()]
    skill_patterns = [nlp.make_doc(skill) for skill in skills]
    matcher.add("SKILL", skill_patterns)
    return matcher, skills

# === Replace the extract_skills function in ALL THREE files (analyzer, recommender, app) with this one ===

def extract_skills(text, matcher, all_skills_list):
    """
    Extracts skills using a hybrid approach with a final filtering step.
    Finds both multi-word phrases and single-word skills from our known list.
    """
    doc = nlp(text.lower())
    found_skills = set()
    
    # 1. Phrase-based matching for multi-word skills from skills.txt
    matches = matcher(doc)
    for _, start, end in matches:
        found_skills.add(doc[start:end].text)
        
    # 2. Token-based matching for single-word skills from skills.txt
    for token in doc:
        # Check if the token's lowercase form is in our master skill list
        if token.text.lower() in all_skills_list:
            found_skills.add(token.text.lower())
            
    return list(found_skills)

def parse_resume_pdf(uploaded_file):
    try:
        # Streamlit's uploaded file is a file-like object, so we read its content
        file_bytes = uploaded_file.read()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def get_recommendations(resume_skills, job_df, top_n=5):
    job_skills_str = job_df['Skills'].apply(lambda skills: ' '.join(skills))
    resume_skills_str = ' '.join(resume_skills)
    vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b')
    job_vectors = vectorizer.fit_transform(job_skills_str)
    resume_vector = vectorizer.transform([resume_skills_str])
    similarity_scores = cosine_similarity(resume_vector, job_vectors)
    scores = similarity_scores[0]
    top_indices = np.argsort(scores)[::-1][:top_n]
    recommendations_df = job_df.iloc[top_indices].copy()
    recommendations_df['Match_Score'] = [round(scores[i] * 100, 2) for i in top_indices] # As percentage
    return recommendations_df[['Title', 'Company', 'Location', 'Match_Score', 'Skills']]

def analyze_skill_gap(resume_skills, recommended_job):
    resume_skills_set = set(resume_skills)
    job_skills_set = set(recommended_job['Skills'])
    missing_skills = list(job_skills_set - resume_skills_set)
    matching_skills = list(job_skills_set.intersection(resume_skills_set))
    return missing_skills, matching_skills


# --- STREAMLIT USER INTERFACE ---

st.title("AI-Powered Job Recommendation & Skill Gap Analyzer")
st.write("Upload your resume and select a job category to find the best job matches and see what skills you need.")

# --- THIS IS THE DYNAMIC PART ---
# Create a dropdown for the user to select the job category.
# We get the unique categories from our DataFrame.
job_categories = job_df['Job_Category'].unique()
selected_category = st.selectbox("1. Select a Job Category", options=job_categories)

# --- File Uploader ---
uploaded_resume = st.file_uploader("2. Upload Your Resume (PDF only)", type=['pdf'])

# --- Analyze Button ---
if st.button("Analyze and Recommend"):
    if uploaded_resume is not None and selected_category:
        with st.spinner('Analyzing your resume and finding matches...'):
            # --- Run the entire pipeline when the button is clicked ---
            
            # 1. Parse Resume
            resume_text = parse_resume_pdf(uploaded_resume)
            
            if resume_text:
                # 2. Extract Skills from Resume
                skill_matcher, all_skills = create_skill_matcher('skills.txt')
                resume_skills = extract_skills(resume_text, skill_matcher, all_skills)
                st.subheader("Skills Found in Your Resume:")
                st.write(", ".join(resume_skills)) # Display skills as a nice string

                # 3. Filter Jobs based on USER'S selection
                filtered_jobs = job_df[job_df['Job_Category'] == selected_category].copy()

                if not filtered_jobs.empty:
                    # 4. Get Recommendations
                    recommendations = get_recommendations(resume_skills, filtered_jobs, top_n=5)

                    st.subheader(f"Top 5 Job Recommendations for '{selected_category}'")
                    st.dataframe(recommendations.style.format({'Match_Score': '{:.2f}%'}))

                    # 5. Skill Gap Analysis for the Top Job
                    if not recommendations.empty:
                        top_job = recommendations.iloc[0]
                        st.subheader(f"Skill Gap Analysis for Top Job: '{top_job['Title']}'")
                        
                        missing, matching = analyze_skill_gap(resume_skills, top_job)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"Matching Skills ({len(matching)}):")
                            st.write(", ".join(matching))
                        with col2:
                            st.warning(f"Missing Skills ({len(missing)}):")
                            st.write(", ".join(missing))
                else:
                    st.error(f"No jobs found for the category: {selected_category}")
    else:
        st.warning("Please upload a resume and select a job category.")