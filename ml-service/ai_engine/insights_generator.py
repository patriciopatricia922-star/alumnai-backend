# ai_engine/insights_generator.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


class InsightsGenerator:
    def __init__(self):
        self.model_data = None
        self._load_model()

    def _load_model(self):
        """Load existing predictive model"""
        try:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.pkl")
            if os.path.exists(model_path):
                self.model_data = joblib.load(model_path)
                print("   ✅ Existing predictive model loaded")
        except Exception as e:
            print(f"   ⚠️ Could not load model: {e}")
            self.model_data = None

    def extract_keywords(self, texts, top_n=10):
        """Extract important keywords from texts"""
        if not texts or len(texts) < 2:
            return []

        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf = vectorizer.fit_transform(texts)

            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf.sum(axis=0).A1

            top_indices = scores.argsort()[-top_n:][::-1]
            return [feature_names[i] for i in top_indices if scores[i] > 0]
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            return []

    def generate_insights(self, feedback_data):
        """Generate insights for dashboard"""
        if not feedback_data:
            return {
                "has_data": False,
                "message": "No feedback data available"
            }

        lengths = [len(f) for f in feedback_data]

        return {
            "has_data": True,
            "total": len(feedback_data),
            "avg_length": round(np.mean(lengths), 1) if lengths else 0,
            "keywords": self.extract_keywords(feedback_data)
        }