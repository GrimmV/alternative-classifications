from pydantic import BaseModel, Field
from typing import List

# Example response models
class ClassificationResponse(BaseModel):
    """Example response model for summarization tasks"""
    emotionality: float = Field(..., description="Emotionality of the text from -1 (very negative) to 1 (very positive)")
    sentiment: str = Field(..., description="Overall sentiment of the text (positive/negative/neutral)")
    polarization: float = Field(..., description="Polarization of the text from 1 (very polarizing) to 0 (very nuanced)")
    sarcasm: float = Field(..., description="Sarcasm of the text from 1 (very sarcastic) to 0 (not sarcastic)")
    claims: List[str] = Field(..., description="List of claims made in the text")
    topics: List[str] = Field(..., description="List of topics discussed in the text")
    societal_relevance: float = Field(..., description="Societal relevance of the text from 1 (very relevant) to 0 (not relevant)")
    fake_news: bool = Field(..., description="Whether the text is fake news")
    fake_news_reason: str = Field(..., description="Reason for the classification")
