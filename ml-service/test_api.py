# test_ai.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_engine import init_ai, sentiment_analyzer, feedback_summarizer, insights_generator

    print("✅ AI engine imported successfully")

    print("Initializing AI...")
    init_ai()

    print("Testing sentiment analysis...")
    result = sentiment_analyzer.analyze("This is a great alumni event!")
    print(f"Sentiment result: {result}")

    print("AI test completed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()