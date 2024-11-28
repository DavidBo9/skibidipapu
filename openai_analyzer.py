# openai_analyzer.py
from flask import jsonify
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class EmotionAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def get_personalized_analysis(self, text_sentiment, text_emotion, face_emotion):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",  # or another suitable model
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are an empathetic AI consultant analyzing emotional states from text and facial expressions.
                        Provide personalized recommendations based on the emotional analysis results.
                        Include:
                        1. A brief analysis of the emotional state
                        2. 2-3 specific activity recommendations
                        3. A positive, encouraging message
                        
                        Keep responses concise but meaningful, about 3-4 sentences.
                        Focus on constructive, helpful suggestions that could improve the person's emotional well-being.
                        """
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Text Analysis Results:
                        - Sentiment: {text_sentiment}
                        - Primary Text Emotion: {text_emotion}
                        - Facial Expression: {face_emotion}
                        
                        Please provide a personalized analysis and recommendations.
                        """
                    }
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating analysis: {str(e)}"

