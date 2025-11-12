import os # to create folders and manage file paths
import requests # to send HTTP requests and download HTML content
from bs4 import BeautifulSoup # to parse and extract links from HTML
import subprocess # to execute shell commands (like git clone)

# ----------------------------
# CONFIGURATION SECTION
# ----------------------------

# The base URL for ENVRI notebook search 
BASE_URL = "https://search.envri.eu/notebookSearch/genericsearch"

# Search term (e.g., "ocean")
# SEARCH_TERM = "ocean"
SEARCH_TERM = "forest"

# Number of pages of search results you want to process
# NUM_PAGES = 21
NUM_PAGES = 43

# My local directory to store download notebooks
output_dir = "/home/nafis/Development/jupyter-quality-extension/11_envri_validation_set"

# Create the directory if it doesn't already exist
os.makedirs(output_dir, exist_ok=True)

# Initialize a counter to keep track of how many notebooks were downloaded
downloaded = 0

# ----------------------------
# MAIN LOOP THROUGH PAGES
# ----------------------------
for page in range (1, NUM_PAGES + 1):
    print(f"Scanning page {page}...")

    # Construct the page URL with the search term and page number
    url = f"{BASE_URL}?term={SEARCH_TERM}&page={page}"

    # Request the HTML content of the page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all <a> tags that contain hyperlinks
    links = soup.find_all("a", href=True)

    # Loop through each hyperlink fuond on the page
    for link in links:
        href = link["href"].strip() # Get the href attribute and clean up whitespace
        #print(href)
        # Check if it's a GitHub link (where the actual notebook code lives)
        if href.startswith("https://github.com/"):
            repo_url = href # This is the GitHub repo URL

            # Extract the repo name from the URL (used to name the folder)
            repo_name = repo_url.rsplit("/", 1)[-1]

            # Define the destination path on my local system
            dest_path = os.path.join(output_dir, repo_name)

            # If the repo already exists locally, skip to avoid duplicate downloads
            if os.path.exists(dest_path):
                print(f"Already exists locally: {repo_name}")
                continue

            # Clone the repo using the `git clone` command
            try:
                subprocess.run(["git", "clone", repo_url, dest_path], check=True)
                print(f"✓ Cloned: {repo_name}")
                downloaded += 1
            except subprocess.CalledProcessError as e:
                print(f"× Failed to clone {repo_url}: {e}")

            #print(repo_url)
    # if downloaded > 1:
    #     break

            

print(f"\n Finished! Total GitHub repositories cloned: {downloaded}")

