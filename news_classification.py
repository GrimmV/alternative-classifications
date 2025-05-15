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
    texts = ["""
    Artificial Intelligence (AI) is transforming various industries. 
    Machine learning algorithms are becoming more sophisticated, 
    enabling better decision-making and automation. 
    However, there are concerns about job displacement and ethical implications.
    """, "Voter ID is supported by an overwhelming majority of NYers, from all across the state, walks of life, & political parties."]
    
    for text in texts:
        # Generate a structured summary
        result = classifier.generate(
            f"Please classify the following text and provide key points: {text}"
        )
    
        print("Statement: ", text)
        
        print("Emotionality:", result.emotionality)
        print("Sentiment:", result.sentiment)
        print("Polarization:", result.polarization)
        print("Sarcasm:", result.sarcasm)
        print("Claims:", result.claims)
        print("Topics:", result.topics)
        print("--------------------------------")
        print("\n")

if __name__ == "__main__":
    main()