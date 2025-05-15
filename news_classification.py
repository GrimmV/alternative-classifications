from llama_model import LlamaModel
from ResponseModels.ClassificationResponse import ClassificationResponse
from typing import Type
import datasets
import pandas as pd
import json
import os

class NewsClassification(LlamaModel[ClassificationResponse]):

    def get_response_model(self) -> Type[ClassificationResponse]:
        return ClassificationResponse


# Example usage
def main():
    
    path = "data/news_classification.csv"
    
        # Create a summarizer instance
    classifier = NewsClassification()
    n_samples = 200
    dataset = "chengxuphd/liar2"
    dataset = datasets.load_dataset(dataset)
    train_raw = pd.DataFrame(dataset["train"])
    
    train_raw = train_raw.sample(n=n_samples)

    # for X_train extract a list of dictionaries with the keys "statement" and "speaker"
    X_train = train_raw[["statement", "speaker"]].to_dict(orient="records")
    # for y_train extract a list of the "label" column
    y_train = train_raw["label"].to_list()
    
    df_dict = {
        "statement": [],
        "speaker": [],
        "emotionality": [],
        "sentiment": [],
        "polarization": [],
        "sarcasm": [],
        "claims": [],
        "topics": [],
        "societal_relevance": [],
        "societal_relevance_reason": [],
        "fake_news": [],
        "fake_news_reason": [],
        "label": []
    }

    for i, text in enumerate(X_train):
        
        print(f"{i}th statement")
        print("Statement: ", text)
        # Generate a structured summary
        result = classifier.generate(
            f"Please analyze the following statement: {json.dumps(text)}"
        )

        df_dict["statement"].append(text["statement"])
        df_dict["speaker"].append(text["speaker"])
        df_dict["emotionality"].append(result.emotionality)
        df_dict["sentiment"].append(result.sentiment)
        df_dict["polarization"].append(result.polarization)
        df_dict["sarcasm"].append(result.sarcasm)
        df_dict["claims"].append(result.claims)
        df_dict["topics"].append(result.topics)
        df_dict["societal_relevance"].append(result.societal_relevance)
        df_dict["societal_relevance_reason"].append(result.societal_relevance_reason)
        df_dict["fake_news"].append(result.fake_news)
        df_dict["fake_news_reason"].append(result.fake_news_reason)
        df_dict["label"].append(False if y_train[i] < 4 else True)
        print("--------------------------------")
        print("\n")

    df = pd.DataFrame(df_dict)
    df.to_csv(path, index=False)
        
    

if __name__ == "__main__":
    main()
