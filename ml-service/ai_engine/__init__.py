# ai_engine/__init__.py
from .sentiment_analyzer import SentimentAnalyzer
from .feedback_summarizer import FeedbackSummarizer
from .insights_generator import InsightsGenerator

# Global variables
sentiment_analyzer = None
feedback_summarizer = None
insights_generator = None


def init_ai():
    """Initialize AI components"""
    global sentiment_analyzer, feedback_summarizer, insights_generator

    print("🔄 Loading AI models...")
    sentiment_analyzer = SentimentAnalyzer()
    feedback_summarizer = FeedbackSummarizer()
    insights_generator = InsightsGenerator()
    print("✅ AI models ready!")

    return True


# Auto-initialize when imported
init_ai()