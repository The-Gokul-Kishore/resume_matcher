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