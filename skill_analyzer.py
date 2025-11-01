import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import re
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. LOAD SPACY MODEL ---
# Load the large English model. We do this once at the start.
print("Loading spaCy model...")
nlp = spacy.load('en_core_web_lg')
print("Model loaded.")


# --- 2. LOAD AND PREPARE DATA ---
def load_and_clean_data(filepath, min_desc_length=50):
    """Loads, cleans, and filters the job data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Initial number of job listings: {len(df)}")
    
    df.dropna(subset=['Description'], inplace=True)
    df = df[df['Description'].astype(str).apply(lambda x: len(x.split()) > min_desc_length)]
    print(f"Listings after filtering: {len(df)}")

    if df.empty:
        print("No valid job listings remaining after filtering.")
        return df

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s#+\.]', '', text)
        return text

    df['Cleaned_Description'] = df['Description'].astype(str).apply(clean_text)
    df.reset_index(drop=True, inplace=True)
    print("Data loaded, filtered, and cleaned.")
    return df


# --- 3. THE CORE AI FUNCTION: SKILL EXTRACTION ---
def create_skill_matcher(skill_file_path):
    """Loads skills from a file and creates a spaCy PhraseMatcher."""
    # --- WHY PhraseMatcher? ---
    # It's highly efficient for finding sequences of words (like "natural language processing").
    # It's much faster than looping through the text and checking for substrings.
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    
    with open(skill_file_path, 'r') as f:
        skills = [line.strip().lower() for line in f if line.strip()]
    
    # Create spaCy pattern documents for each skill
    skill_patterns = [nlp.make_doc(skill) for skill in skills]
    matcher.add("SKILL", skill_patterns)
    return matcher, skills # Return the list of skills as well for later use

# === Replace the old extract_skills function in BOTH files with this one ===

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

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Define file paths
    job_data_filepath = 'master_google_jobs.csv'
    skills_filepath = 'skills.txt'
    
    # Load and clean data
    job_df = load_and_clean_data(job_data_filepath)
    
    if not job_df.empty:
        # Create the skill matcher
        skill_matcher, all_skills = create_skill_matcher(skills_filepath)
        
        # --- APPLY THE SKILL EXTRACTOR ---
        # We create a new 'Skills' column by applying our AI function to each job description.
        # This is the most important step of this phase.
        print("\nExtracting skills from job descriptions... (this might take a few minutes)")
        job_df['Skills'] = job_df['Cleaned_Description'].apply(lambda text: extract_skills(text, skill_matcher, all_skills))
        print("Skill extraction complete.")
        
        # Save the processed data to a new CSV for future use
        # This is a crucial step! We don't want to re-run skill extraction every time.
        processed_data_filepath = 'jobs_with_skills.csv'
        job_df.to_csv(processed_data_filepath, index=False)
        print(f"Data with skills saved to {processed_data_filepath}")

        # --- Let's inspect the results ---
        print("\n--- DataFrame with new 'Skills' column ---")
        # We'll display relevant columns: Title, Company, Job_Category, and the new Skills
        print(job_df[['Title', 'Job_Category', 'Skills']].head())
        
        # Let's see the most common skills for a specific job category
        print("\n--- Top 10 Most Common Skills for 'Data Scientist' ---")
        ds_skills = job_df[job_df['Job_Category'] == 'Data Scientist']['Skills'].explode()
        common_skills = Counter(ds_skills).most_common(10)
        print(pd.DataFrame(common_skills, columns=['Skill', 'Count']))