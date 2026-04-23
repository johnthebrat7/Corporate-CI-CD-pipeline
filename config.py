import os

class Config:
    # Flask Security
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev_secret_key_123")
    DEBUG = os.environ.get("DEBUG", "True").lower() == "true"
    
    # Directory Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # ML Artifacts (The only two files you have)
    VECTORIZER_PATH = os.path.join(MODEL_DIR, 'countVectorizer.pkl')
    MODEL_PATH = os.path.join(MODEL_DIR, 'model_xgb.pkl')
    
    # Input Constraints
    MIN_TEXT_LENGTH = 3
    MAX_TEXT_LENGTH = 5000