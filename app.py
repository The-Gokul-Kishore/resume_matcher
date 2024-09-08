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

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_number in range(len(reader.pages)):
            text += reader.pages[page_number].extract_text()
    return text

def process_resumes(job_description, limit):
    # Check if cache exists
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            cached_data = pickle.load(f)
        resumes = cached_data['resumes']
        resume_details = cached_data['resume_details']
    else:
        resumes = []
        resume_details = []
        files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith(".pdf")]
        total_files = len(files)

        # Process each resume file
        for i, filename in enumerate(files):
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

    # Vectorize job description and resumes using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = [job_description] + resumes
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Calculate similarity between job description and resumes
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    # Sort resumes by similarity score in descending order
    sorted_indices = similarity_scores.argsort()[0][::-1]
    sorted_resumes = [resume_details[i] for i in sorted_indices[:limit]]

    # Add match percentage to the sorted resumes
    for i, idx in enumerate(sorted_indices[:limit]):
        sorted_resumes[i]["match_percentage"] = round(similarity_scores[0][idx] * 100, 2)

    return sorted_resumes

@app.route('/api/progress')
def progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f).get("progress", 0)
        return jsonify({"progress": progress})
    return jsonify({"progress": 0})

@app.route('/api/process', methods=['POST'])
def process():
    data = request.json
    job_description = data.get('job_description')
    limit = data.get('limit')

    # Clear previous cache if new processing is requested
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    matching_resumes = process_resumes(job_description, limit)
    return jsonify(matching_resumes)

@app.route('/api/stop', methods=['POST'])
def stop_processing():
    global processing_flag
    processing_flag = True
    return jsonify({"message": "Processing stopped"}), 200

if __name__ == "__main__":
    app.run(debug=True)
