import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def analyze_news_classification(df: pd.DataFrame):
    # plot a bar chart of the number of fake news and real news
    plt.bar(["Fake News", "Real News"], [df["label"].value_counts()[1], df["label"].value_counts()[0]])
    plt.show()

def analyze_classification_metrics(df: pd.DataFrame):
    
    # generate the confusion matrix
    cm = confusion_matrix(df["label"], ~df["fake_news"])
    
    # visualize the confusion matrix
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()
    
    
# Scatterplot that shows the relationship between sarcasm and polarization colored by fake news
def analyze_sarcasm_polarization(df: pd.DataFrame):
    plt.scatter(df["sarcasm"], df["polarization"], c=df["fake_news"])
    plt.show()
    
    
