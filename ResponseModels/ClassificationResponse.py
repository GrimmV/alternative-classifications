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
    societal_relevance: float = Field(..., description="Societal relevance of the text from 1 (very relevant) to 0 (not relevant). Take into account the extracted claims and topics.")
    societal_relevance_reason: str = Field(..., description="The reason the text is or is not relevant to society.")
    fake_news: bool = Field(..., description="Whether the text is fake news, based on the extracted information above and the given text.")
    fake_news_reason: str = Field(..., description="The reason the text is or is not fake news. Your reasoning should be based on the extracted information above and the given text.")
