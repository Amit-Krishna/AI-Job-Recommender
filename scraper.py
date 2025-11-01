import os
from serpapi import GoogleSearch
import pandas as pd
import time

# --- IMPORTANT ---
# Make sure your SerpApi API Key is pasted here.
SERPAPI_API_KEY = "6299b483934e8db81c797db4e763533b3e279038ca8928d3b93797ef16fe2ecd" 

def scrape_single_job_type(job_title, location, pages=1):
    """
    Scrapes a single job type from Google Jobs using SerpApi.
    This function is now a helper for the main scraping process.
    """
    print(f"\n--- Starting scrape for: {job_title} in {location} ---")
    
    all_job_results = []
    page_count = 0
    params = {
        "engine": "google_jobs",
        "q": job_title,
        "location": location,
        "api_key": SERPAPI_API_KEY
    }

    while True:
        page_count += 1
        print(f"Scraping page {page_count}...")
        
        search = GoogleSearch(params)
        results = search.get_dict()

        if 'error' in results:
            print(f"An error occurred: {results['error']}")
            break
        
        if 'jobs_results' not in results or not results['jobs_results']:
            print("No more job results found.")
            break

        for job in results['jobs_results']:
            # --- NEW FEATURE ---
            # We add a 'Job_Category' column so we can filter by it later in the app.
            all_job_results.append({
                'Job_Category': job_title, # Add the search term as a category
                'Title': job.get('title', 'N/A'),
                'Company': job.get('company_name', 'N/A'),
                'Location': job.get('location', 'N/A'),
                'Description': job.get('description', 'N/A'),
                'Via': job.get('via', 'N/A'),
            })

        if 'serpapi_pagination' not in results or 'next_page_token' not in results['serpapi_pagination']:
            print("Reached the last page of results.")
            break
        
        if page_count >= pages:
            print(f"Finished scraping the requested {pages} page(s).")
            break

        params['next_page_token'] = results['serpapi_pagination']['next_page_token']
        time.sleep(1) # Be respectful to the API

    return all_job_results

# --- MAIN EXECUTION BLOCK for MULTI-JOB SCRAPING ---
if __name__ == "__main__":
    # --- DEFINE THE JOBS YOU WANT TO SCRAPE ---
    # This list will be the options in your app's dropdown menu.
    JOBS_TO_SCRAPE = [
        "Data Scientist",
        "Data Analyst",
        "Software Engineer",
        "Product Manager",
        "UX Designer"
    ]
    
    LOCATION_TO_SEARCH = "United States"
    PAGES_PER_JOB = 5  # Scrape 5 pages (~50 jobs) for each job title

    master_job_list = []
    
    for job_title in JOBS_TO_SCRAPE:
        job_results = scrape_single_job_type(job_title, LOCATION_TO_SEARCH, PAGES_PER_JOB)
        if job_results:
            master_job_list.extend(job_results)
        time.sleep(5) # Wait 5 seconds between different job title scrapes

    if master_job_list:
        # Create a single DataFrame with all the jobs
        master_df = pd.DataFrame(master_job_list)
        
        # Save to a single master CSV file
        output_filename = 'master_google_jobs.csv'
        master_df.to_csv(output_filename, index=False)
        
        print(f"\n--- SCRAPING COMPLETE ---")
        print(f"Data for all jobs saved to {output_filename}")
        print(f"Total jobs scraped: {len(master_df)}")
        
        # Display the counts for each job category to verify
        print("\n--- Job Counts per Category ---")
        print(master_df['Job_Category'].value_counts())
    else:
        print("\nNo job data was scraped. Check your API key and search parameters.")