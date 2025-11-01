import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. LOAD MODELS AND DATA ---
print("Loading spaCy model...")
nlp = spacy.load('en_core_web_lg')
print("Model loaded.")

PROCESSED_DATA_PATH = 'jobs_with_skills.csv'
print(f"Loading processed job data from {PROCESSED_DATA_PATH}...")
job_df = pd.read_csv(PROCESSED_DATA_PATH)
job_df['Skills'] = job_df['Skills'].apply(lambda x: eval(x) if isinstance(x, str) else x)
print("Job data loaded.")


# --- 2. SKILL EXTRACTION FUNCTIONS ---
def create_skill_matcher(skill_file_path):
    """Loads skills from a file and creates a spaCy PhraseMatcher."""
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


# --- 3. RESUME PARSING FUNCTION ---
def parse_resume_pdf(pdf_path):
    """Parses a PDF file and extracts its text content."""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

# --- 4. NEW FUNCTION: RECOMMENDATION LOGIC ---
def get_recommendations(resume_skills, job_df, top_n=5):
    """
    Calculates cosine similarity and returns top N job recommendations.
    """
    # --- WHY DO WE JOIN SKILLS INTO A STRING? ---
    # Scikit-learn's vectorizers work on text. So we convert our list of skills
    # back into a single string for each job and the resume.
    job_skills_str = job_df['Skills'].apply(lambda skills: ' '.join(skills))
    resume_skills_str = ' '.join(resume_skills)

    # --- TF-IDF VECTORIZATION ---
    # This converts our text into numerical vectors. TF-IDF is great because it
    # gives more weight to rare, important skills and less weight to very common ones.
    vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b') # Simple tokenizer
    
    # Create the vocabulary from job skills and transform them into vectors
    job_vectors = vectorizer.fit_transform(job_skills_str)
    
    # Transform the resume skills into a vector using the SAME vocabulary
    resume_vector = vectorizer.transform([resume_skills_str])

    # --- COSINE SIMILARITY CALCULATION ---
    # This computes the similarity between the single resume vector and all job vectors.
    similarity_scores = cosine_similarity(resume_vector, job_vectors)

    # --- RANKING AND FORMATTING ---
    # We get the scores for the first (and only) resume
    scores = similarity_scores[0]
    
    # Get the indices of the top N highest scores
    top_indices = np.argsort(scores)[::-1][:top_n]
    
    # Create a DataFrame with the recommendations
    recommendations_df = job_df.iloc[top_indices].copy()
    recommendations_df['Match_Score'] = [scores[i] for i in top_indices]
    
    return recommendations_df[['Title', 'Company', 'Location', 'Job_Category', 'Match_Score', 'Skills']]


# --- 5. NEW FUNCTION: SKILL GAP ANALYSIS ---
def analyze_skill_gap(resume_skills, recommended_job):
    """
    Compares resume skills with the skills of a recommended job to find the gap.
    """
    resume_skills_set = set(resume_skills)
    job_skills_set = set(recommended_job['Skills'])
    
    missing_skills = list(job_skills_set - resume_skills_set)
    matching_skills = list(job_skills_set.intersection(resume_skills_set))
    
    return missing_skills, matching_skills


# --- 6. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    skills_filepath = 'skills.txt'
    resume_filepath = 'sample_resume.pdf'

    # --- Step A: Process the Resume ---
    print("\n--- Processing Resume ---")
    resume_text = parse_resume_pdf(resume_filepath)
    if resume_text:
        skill_matcher, all_skills = create_skill_matcher(skills_filepath)
        resume_skills = extract_skills(resume_text, skill_matcher, all_skills)
        print(f"Skills found in resume: {resume_skills}")

        # --- Step B: Filter Jobs (for the future web app) ---
        # Here, the user would select a job category. We'll hardcode it for now.
        USER_JOB_CHOICE = "Data Analyst"
        print(f"\n--- Filtering for '{USER_JOB_CHOICE}' jobs ---")
        filtered_jobs = job_df[job_df['Job_Category'].str.contains(USER_JOB_CHOICE, case=False, na=False)].copy()

        if not filtered_jobs.empty:
            # --- Step C: Get Recommendations ---
            print("\n--- Getting Top 5 Job Recommendations ---")
            recommendations = get_recommendations(resume_skills, filtered_jobs, top_n=5)
            print(recommendations.to_string()) # .to_string() prints the full DataFrame

            # --- Step D: Analyze Skill Gap for the Top Job ---
            if not recommendations.empty:
                top_job = recommendations.iloc[0]
                print(f"\n--- Skill Gap Analysis for Top Job: '{top_job['Title']}' ---")
                
                missing, matching = analyze_skill_gap(resume_skills, top_job)
                
                print(f"Matching Skills ({len(matching)}): {matching}")
                print(f"Missing Skills ({len(missing)}): {missing}")
        else:
            print(f"No jobs found for the category: {USER_JOB_CHOICE}")
    else:
        print(f"Could not process resume: {resume_filepath}")