from llama_model import LlamaModel
from ResponseModels.ClassificationResponse import ClassificationResponse
from typing import Type

class NewsClassification(LlamaModel[ClassificationResponse]):

    def get_response_model(self) -> Type[ClassificationResponse]:
        return ClassificationResponse
    
    
