from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Optional
import requests
from pydantic import BaseModel
import instructor
from openai import OpenAI

T = TypeVar('T', bound=BaseModel)

class LlamaModel(ABC, Generic[T]):
    """Abstract base class for Llama model interactions"""
    
    def __init__(self, model_name: str = "llama3:70b", temperature: float = 0.7):
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
