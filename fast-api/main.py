from fastapi import FastAPI
import uvicorn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from scipy.special import softmax
import torch
import numpy as np
import urllib.request
import csv

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
config = AutoConfig.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

#moving the model to cuda
model.to('cuda')
print("Model Loaded Successfully")


app = FastAPI()

@app.get("/textclassification")
# def healthcheck():
#     return 'Health - OK'

def predict_sentiment(text: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    #Encoding input text
    encoded_input = tokenizer(text, return_tensors='pt')
    input_ids = encoded_input.to('cuda')
    output = model(**input_ids)
    
    # perform prediction
    scores = output[0][0].detach().cpu().numpy()
    scores = softmax(scores)
    
    task='sentiment'
    #labels 
    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]
    
    
    
    # output
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    temp = []
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        res = {"prediction":{l}, "Probability":{np.round(float(s), 4)}}
        temp.append(res)
    return temp


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = 8080)