# AI-Powered Job Recommender & Skill Gap Analyzer

An end-to-end data science application that helps users find relevant job postings and identify skill gaps. The tool scrapes job data, uses Natural Language Processing (NLP) to extract required skills, and compares them against a user's uploaded PDF resume to provide personalized recommendations and an actionable skill gap analysis.

## Features

* **Dynamic Job Selection:** Users can select from multiple pre-loaded job categories (e.g., Data Scientist, Software Engineer, UX Designer).
* **Resume Parsing:** Automatically extracts text and skills from an uploaded PDF resume.
* **AI-Powered Recommendations:** Uses a TF-IDF Vectorizer and Cosine Similarity to calculate a match score and rank jobs based on skill overlap.
* **Skill Gap Analysis:** Clearly shows which skills the user has ("Matching Skills") and which they are missing ("Missing Skills") for the top-ranked job.
* **Interactive Web Interface:** Built with Streamlit for a clean, user-friendly experience.

## Tech Stack

* **Data Acquisition:** `SerpApi` for reliable scraping of Google Jobs.
* **Data Processing:** `Pandas`, `NumPy`.
* **NLP & AI:** `spaCy` for linguistic processing and `Scikit-learn` for TF-IDF vectorization and cosine similarity.
* **PDF Parsing:** `PyMuPDF`.
* **Web Framework / UI:** `Streamlit`.

## How to Run Locally

Follow these steps to set up and run the project on your own machine.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Job-Recommender.git
cd AI-Job-Recommender
# Replace your-username with your GitHub username
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model

The project's NLP component requires the large English model from spaCy.

```bash
python -m spacy download en_core_web_lg
```

### 5. Launch the Application

Run the Streamlit app from your terminal.

```bash
streamlit run app.py
```

Your web browser should automatically open a new tab with the application running.

## Project Structure

```
.
├── app.py                  # The main Streamlit web application
├── recommender.py          # Backend logic for command-line testing
├── skill_analyzer.py       # Script to process data and extract skills
├── google_jobs_scraper.py  # Script for scraping job data via SerpApi
├── requirements.txt        # List of all Python dependencies
├── skills.txt              # The master dictionary of skills for the NLP model
├── .gitignore              # Specifies files for Git to ignore
├── master_google_jobs.csv  # The raw scraped data file
├── jobs_with_skills.csv    # The processed and enriched data file used by the app
└── sample_resume.pdf       # A sample resume for testing the application
```
