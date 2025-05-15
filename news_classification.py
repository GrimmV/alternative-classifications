from llama_model import LlamaModel
from ResponseModels.ClassificationResponse import ClassificationResponse
from typing import Type

class NewsClassification(LlamaModel[ClassificationResponse]):
    
    def get_response_model(self) -> Type[ClassificationResponse]:
        return ClassificationResponse

# Example usage
def main():
    # Create a summarizer instance
    classifier = NewsClassification()
    
    # Example text to summarize
    text = """
    Artificial Intelligence (AI) is transforming various industries. 
    Machine learning algorithms are becoming more sophisticated, 
    enabling better decision-making and automation. 
    However, there are concerns about job displacement and ethical implications.
    """
    
    # Generate a structured summary
    result = classifier.generate(
        f"Please classify the following text and provide key points: {text}"
    )
    
    print("Emotionality:", result.emotionality)
    print("Sentiment:", result.sentiment)
    print("Polarization:", result.polarization)
    print("Sarcasm:", result.sarcasm)
    print("Claims:", result.claims)
    print("Topics:", result.topics)

if __name__ == "__main__":
    main()