# ai_engine/feedback_summarizer.py
from transformers import pipeline
import numpy as np


class FeedbackSummarizer:
    def __init__(self):
        self.summarizer = None
        self._load_model()

    def _load_model(self):
        """Load text generation model for summarization"""
        try:
            print("   Loading summarization model...")
            self.summarizer = pipeline(
                "text-generation",
                model="gpt2",
                device=-1
            )
            print("   ✅ Summarization model ready (using text generation)")
        except Exception as e:
            print(f"   ⚠️ Could not load summarizer: {e}")
            self.summarizer = None

    def summarize(self, texts, max_length=150):
        """Summarize feedback texts"""
        if not self.summarizer or not texts:
            return self._fallback_summary(texts)

        try:
            combined = " | ".join(texts[:5])  # Take first 5 to keep it short
            if len(combined) > 500:
                combined = combined[:500]

            prompt = f"Summarize the following alumni feedback: {combined}"

            result = self.summarizer(
                prompt,
                max_new_tokens=80,
                temperature=0.7,
                do_sample=True,
                truncation=True
            )

            summary = result[0]['generated_text']
            summary = summary.replace(prompt, "").strip()

            if len(summary) < 20:
                return self._fallback_summary(texts)

            return summary
        except Exception as e:
            print(f"Summarization error: {e}")
            return self._fallback_summary(texts)

    def _fallback_summary(self, texts):
        """Fallback summary when model fails"""
        if not texts:
            return "No feedback available to summarize."

        total = len(texts)
        avg_length = sum(len(t) for t in texts) // total if total > 0 else 0

        return f"Based on {total} feedback entries (average length: {avg_length} characters), alumni have shared their experiences and suggestions."

    def extract_themes(self, texts, top_n=5):
        """Extract common themes from feedback"""
        common_keywords = {
            "curriculum": ["course", "subject", "curriculum", "program", "major", "class", "academic"],
            "facilities": ["facility", "lab", "classroom", "equipment", "library", "building", "campus"],
            "career": ["job", "career", "employment", "internship", "placement", "work", "profession"],
            "events": ["event", "activity", "seminar", "webinar", "workshop", "networking", "gathering"],
            "faculty": ["teacher", "professor", "faculty", "instructor", "mentor", "staff"],
            "administration": ["admin", "office", "registration", "enrollment", "payment", "records"],
            "online": ["online", "virtual", "zoom", "remote", "platform", "digital"],
            "scholarship": ["scholarship", "financial", "aid", "tuition", "fee", "cost"],
            "alumni": ["alumni", "graduate", "batch", "classmate", "network"],
            "satisfaction": ["satisfied", "happy", "good", "great", "excellent", "enjoy"]
        }

        all_text = " ".join(texts).lower()
        themes = []

        for theme, keywords in common_keywords.items():
            count = 0
            for keyword in keywords:
                count += all_text.count(keyword)
            if count > 0:
                themes.append({"theme": theme, "count": count})

        themes.sort(key=lambda x: x["count"], reverse=True)
        return themes[:top_n]