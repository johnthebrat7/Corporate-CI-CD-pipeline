import os
import pickle
import re
import numpy as np
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from config import Config  # Importing your updated config

app = Flask(__name__)
app.config.from_object(Config)

# --- PREPROCESSING SETUP ---
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    import nltk
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))

stemmer = PorterStemmer()

# --- LOAD ARTIFACTS ---
# Using the paths defined in config.py
try:
    with open(app.config['VECTORIZER_PATH'], 'rb') as f:
        cv = pickle.load(f)
    with open(app.config['MODEL_PATH'], 'rb') as f:
        model = pickle.load(f)
    print("✅ Successfully loaded CountVectorizer and XGBoost Model.")
except FileNotFoundError as e:
    print(f"Error: Could not find model files. Check path: {e.filename}")
    cv, model = None, None

def preprocess_text(text):
    """Standardized preprocessing matching training logic."""
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return ' '.join(review)

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    
    if request.is_json:
        data = request.get_json()
        raw_text = data.get('review') or data.get('text', "")
    else:
        raw_text = request.form.get('review') or request.form.get('text', "")

    
    if not raw_text or len(raw_text) < app.config['MIN_TEXT_LENGTH']:
        error_msg = f"Please enter at least {app.config['MIN_TEXT_LENGTH']} characters."
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        flash(error_msg, "warning")
        return redirect(url_for('home'))

    try:
        # --- STEP 3: Processing & Prediction ---
        processed_text = preprocess_text(raw_text)
        vectorized_input = cv.transform([processed_text]).toarray()

        prediction = model.predict(vectorized_input)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"

        # Confidence Calculation
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            prob_matrix = model.predict_proba(vectorized_input)[0]
            confidence = round(float(np.max(prob_matrix)) * 100, 2)

        # --- STEP 4: Return Response ---
        response = {
            'input_text': raw_text,
            'sentiment': sentiment,
            'confidence': f"{confidence}%",
            'model_used': 'XGBoost'
        }

        if request.is_json:
            return jsonify(response)

        return render_template("result.html", 
                               text=raw_text, 
                               prediction=sentiment, 
                               confidence=confidence)

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        if request.is_json:
            return jsonify({'error': 'Internal processing error'}), 500
        flash("An error occurred during prediction.", "error")
        return redirect(url_for('home'))
    
@app.route("/about")
def about():
    """About page with project information."""
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], port=5000)