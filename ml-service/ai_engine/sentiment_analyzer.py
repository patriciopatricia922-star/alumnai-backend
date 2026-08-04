# ai_engine/sentiment_analyzer.py
from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load transformer model for sentiment analysis"""
        try:
            print("   Loading sentiment analysis model...")
            self.model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # Use CPU
            )
            print("   ✅ Sentiment model ready")
        except Exception as e:
            print(f"   ⚠️ Could not load sentiment model: {e}")
            self.model = None

    def analyze(self, text):
        """Analyze sentiment of feedback text"""
        if self.model is None:
            return {"label": "neutral", "score": 0.5}

        try:
            result = self.model(text[:512])[0]
            return {
                "label": result['label'].lower(),
                "score": float(result['score'])
            }
        except:
            return {"label": "neutral", "score": 0.5}

    def analyze_batch(self, texts):
        """Analyze multiple feedback texts"""
        results = []
        for text in texts:
            if text and len(text) > 10:
                results.append(self.analyze(text))
        return results