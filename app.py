from flask import Flask, request, jsonify, url_for
import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

RESUME_FOLDER = os.getenv("RESUME_FOLDER", "static/resumes/")
PROGRESS_FILE = 'progress.json'
CACHE_FILE = 'resume_cache.pkl'
processing_flag = False  # Global flag to control processing state

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_number in range(len(reader.pages)):
            text += reader.pages[page_number].extract_text()
    return text
from sentence_transformers import SentenceTransformer


def process_resumes(job_description, limit, priority_keywords=None):
    global processing_flag
    resumes = []
    resume_details = []
    
    # Check if cache exists
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            cached_data = pickle.load(f)
        resumes = cached_data['resumes']
        resume_details = cached_data['resume_details']
    else:
        files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith(".pdf")]
        total_files = len(files)

        # Process each resume file
        for i, filename in enumerate(files):
            if processing_flag:  # Check if processing should stop
                break
            
            resume_text = extract_text_from_pdf(os.path.join(RESUME_FOLDER, filename))
            resumes.append(resume_text)
            application_no = os.path.splitext(filename)[0]
            resume_details.append({
                "application_no": application_no,
                "resume_link": url_for('static', filename=f'resumes/{filename}', _external=True)
            })

            # Update progress
            progress = (i + 1) / total_files * 100
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({"progress": progress}, f)

        # Cache the processed resumes
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({'resumes': resumes, 'resume_details': resume_details}, f)

    # Check if processing has been stopped
    if processing_flag:
        return []

    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # A smaller model for efficiency

    # Generate embeddings for job description and resumes
    job_description_embedding = model.encode(job_description)

    # Enhance job description embedding with priority keywords
    if priority_keywords:
        enhanced_job_description = f"{job_description} {priority_keywords}"
        enhanced_job_description_embedding = model.encode(enhanced_job_description)
    else:
        enhanced_job_description_embedding = job_description_embedding

    resume_embeddings = model.encode(resumes)

    # Calculate cosine similarity
    similarity_scores = cosine_similarity([enhanced_job_description_embedding], resume_embeddings)

    # Sort resumes by similarity score in descending order
    sorted_indices = similarity_scores.argsort()[0][::-1]
    sorted_resumes = [resume_details[i] for i in sorted_indices[:limit]]

    # Add match percentage to the sorted resumes
    for i, idx in enumerate(sorted_indices[:limit]):
        sorted_resumes[i]["match_percentage"] = round(similarity_scores[0][idx] * 100, 2)

    return sorted_resumes


@app.route('/api/progress', methods=['GET'])
def progress():
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                progress_data = json.load(f)
            # Safely extract progress, default to 0 if not found
            progress = progress_data.get("progress", 0)
            return jsonify({"progress": progress}), 200
        else:
            return jsonify({"progress": 0}), 200
    except Exception as e:
        # Log the error for debugging
        print(f"Error fetching progress: {e}")
        return jsonify({"error": "Failed to retrieve progress"}), 500
@app.route('/api/process', methods=['POST'])
def process():
    global processing_flag
    processing_flag = False  # Reset the processing flag
    data = request.json
    job_description = data.get('job_description')
    limit = data.get('limit')
    priority_keywords = data.get('priority_keywords')

    # Clear previous cache if new processing is requested
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    matching_resumes = process_resumes(job_description, limit, priority_keywords)
    
    # Ensure the processing flag is correctly set
    processing_flag = False  # Reset flag to allow new processing
    return jsonify(matching_resumes)

@app.route('/api/stop', methods=['POST'])
def stop_processing():
    global processing_flag
    processing_flag = True  # Set the processing flag to stop
    return jsonify({"message": "Processing stopped"}), 200

if __name__ == "__main__":
    app.run(debug=True)
