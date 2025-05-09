from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Optional, List
import requests
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

T = TypeVar('T', bound=BaseModel)

class BaseResponseModel(BaseModel):
    """Base class for all response models"""
    pass

class LlamaModel(ABC, Generic[T]):
    """Abstract base class for Llama model interactions"""
    
    def __init__(self, model_name: str = "llama2:70b", temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,
        )
    
    @abstractmethod
    def get_response_model(self) -> Type[T]:
        """Return the response model class for this model"""
        pass
    
    def generate(self, prompt: str, response_model: Optional[Type[T]] = None) -> T:
        """
        Generate a response using the specified response model
        
        Args:
            prompt (str): The input prompt
            response_model (Type[T], optional): Override the default response model
            
        Returns:
            T: The structured response
        """
        if response_model is None:
            response_model = self.get_response_model()
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                response_model=response_model,
            )
            return response
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error making request: {e}")

# Example response models
class SummaryResponse(BaseResponseModel):
    """Example response model for summarization tasks"""
    summary: str = Field(..., description="A concise summary of the input text")
    key_points: List[str] = Field(..., description="List of key points from the text")
    sentiment: str = Field(..., description="Overall sentiment of the text (positive/negative/neutral)")

class QAResponse(BaseResponseModel):
    """Example response model for question-answering tasks"""
    answer: str = Field(..., description="Direct answer to the question")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    sources: List[str] = Field(default_factory=list, description="List of sources or references used")

# Example implementation
class LlamaSummarizer(LlamaModel[SummaryResponse]):
    """Example implementation for summarization tasks"""
    
    def get_response_model(self) -> Type[SummaryResponse]:
        return SummaryResponse

class LlamaQA(LlamaModel[QAResponse]):
    """Example implementation for question-answering tasks"""
    
    def get_response_model(self) -> Type[QAResponse]:
        return QAResponse

# Example usage
def main():
    # Create a summarizer instance
    summarizer = LlamaSummarizer()
    
    # Example text to summarize
    text = """
    Artificial Intelligence (AI) is transforming various industries. 
    Machine learning algorithms are becoming more sophisticated, 
    enabling better decision-making and automation. 
    However, there are concerns about job displacement and ethical implications.
    """
    
    # Generate a structured summary
    result = summarizer.generate(
        f"Please summarize the following text and provide key points: {text}"
    )
    
    print("Summary:", result.summary)
    print("Key Points:", result.key_points)
    print("Sentiment:", result.sentiment)
    
    # Create a QA instance
    qa = LlamaQA()
    
    # Example question
    question = "What are the main concerns about AI?"
    
    # Generate a structured answer
    answer = qa.generate(question)
    
    print("\nAnswer:", answer.answer)
    print("Confidence:", answer.confidence)
    print("Sources:", answer.sources)

if __name__ == "__main__":
    main() 