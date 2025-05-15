from llm_models.ResponseModels.ClassificationResponse import ClassificationResponse

def content_feedback(result: ClassificationResponse):
    if result.emotionality < - 0.5:
        print("The statement seems to be emotional with a rather negative tone. Is this your intention?")

    if result.polarization > 0.5:
        print("The statement seems to be rather polarized. Is this your intention?")

    if result.sarcasm > 0.5:
        print("The statement seems to be sarcastic. Is this your intention?")

    if result.societal_relevance > 0.5:
        print("The statement seems to be rather societal relevant. Therefore, I advice you to be more careful with your choice of words.")

    if result.fake_news:
        print("The statement seems to be fake news. Is this your intention?")
        
def print_content_feedback(result: ClassificationResponse):
    print("--------------------------------")
    # For each key in the result, print the key and the value
    for key, value in result.model_dump().items():
        print(f"{key}: {value}")
    print("--------------------------------")
    
    
    
