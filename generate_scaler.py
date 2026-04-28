"""
One-time script to regenerate the MinMaxScaler that was used during training.
Run this from the project root: python generate_scaler.py
"""
import os
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Download stopwords if needed
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'amazon_alexa.tsv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CV_PATH   = os.path.join(MODEL_DIR, 'countVectorizer.pkl')
SCALER_OUT = os.path.join(MODEL_DIR, 'scaler.pkl')

# 1. Load data exactly as the notebook did
data = pd.read_csv(DATA_PATH, delimiter='\t', quoting=3)
data.dropna(inplace=True)
print(f"Dataset shape after dropna: {data.shape}")

# 2. Rebuild the corpus with the SAME preprocessing as the notebook
stemmer = PorterStemmer()
corpus = []
for i in range(data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['verified_reviews'])
    review = review.lower().split()
    review = [stemmer.stem(w) for w in review if w not in STOPWORDS]
    corpus.append(' '.join(review))

# 3. Load the SAVED vectorizer and transform (do NOT refit — vocab must match the model)
with open(CV_PATH, 'rb') as f:
    cv = pickle.load(f)
X = cv.transform(corpus).toarray()
y = data['feedback'].values
print(f"X shape: {X.shape}")

# 4. Same split as the notebook (random_state=15 is the key)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15
)

# 5. Fit MinMaxScaler on X_train and save
scaler = MinMaxScaler()
scaler.fit(X_train)

os.makedirs(MODEL_DIR, exist_ok=True)
with open(SCALER_OUT, 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Saved scaler to: {SCALER_OUT}")
print(f"   Scaler data_min sample: {scaler.data_min_[:5]}")
print(f"   Scaler data_max sample: {scaler.data_max_[:5]}")