from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)


MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
log_reg = joblib.load(os.path.join(MODELS_DIR, 'log_reg.pkl'))
svm = joblib.load(os.path.join(MODELS_DIR, 'svm.pkl'))
mlp = joblib.load(os.path.join(MODELS_DIR, 'mlp.pkl'))
meta_model = load_model(os.path.join(MODELS_DIR, 'stacked_model.keras'))
tfidf = joblib.load(os.path.join(MODELS_DIR, 'vectorizer.pkl'))

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    email = request.json['email']
    
    
    email_tfidf = tfidf.transform([email])
    
    
    log_reg_proba = log_reg.predict_proba(email_tfidf)[:, 1]
    svm_proba = svm.predict_proba(email_tfidf)[:, 1]
    mlp_proba = mlp.predict_proba(email_tfidf)[:, 1]
    
    
    stacked_input = np.column_stack([log_reg_proba, svm_proba, mlp_proba])
    prediction = (meta_model.predict(stacked_input) > 0.5).astype(int)[0][0]
    
    return jsonify({'result': 'Spam' if prediction == 1 else 'Not Spam'})

if __name__ == '__main__':
    app.run(debug=True)