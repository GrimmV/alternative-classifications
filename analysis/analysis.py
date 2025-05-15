import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import numpy as np

def analyze_news_classification(df: pd.DataFrame):
    # plot a bar chart of the number of fake news and real news
    plt.bar(["Fake News", "Real News"], [df["label"].value_counts()[1], df["label"].value_counts()[0]])
    plt.show()

def analyze_classification_metrics(df: pd.DataFrame):
    
    # generate the confusion matrix
    cm = confusion_matrix(df["label"], df["fake_news"])
    
    # visualize the confusion matrix
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()
    
    
# Scatterplot that shows the relationship between two numeric columns colored by another numeric column
def analyze_numeric_columns(df: pd.DataFrame, column1: str, column2: str, color_column: str):
    # adapt the points position a little bit so that they are not on top of each other
    df[column1] = df[column1] + np.random.normal(0, 0.01, len(df))
    df[column2] = df[column2] + np.random.normal(0, 0.01, len(df))
    plt.scatter(df[column1], df[column2], c=df[color_column])
    # Add a legend
    plt.legend()
    # Add axis labels
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()
    
    
# visualize a correlation matrix
def analyze_correlation_matrix(df: pd.DataFrame):
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.show()
    

# Calculate performance metrics
def calculate_performance_metrics(df: pd.DataFrame):
    # Calculate precision, recall, F1 score, and accuracy
    precision = precision_score(df["label"], df["fake_news"])
    recall = recall_score(df["label"], df["fake_news"])
    f1 = f1_score(df["label"], df["fake_news"])
    accuracy = accuracy_score(df["label"], df["fake_news"])
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}
