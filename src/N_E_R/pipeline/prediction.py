from N_E_R.config.configuration import ConfigurationManager
# from N_E_R import AutoTokenizer
from N_E_R import pipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertTokenizerFast,pipeline
from datasets import load_dataset, load_from_disk
import torch
import csv
import pandas as pd
import os
import ast
from tqdm import tqdm
import json

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()


    
    def predict(self,text):
        
        label_list =['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']  # Update this list as per your dataset

      
        id2label = {
        str(i): label for i,label in enumerate(label_list)
        }
        label2id = {
        label: str(i) for i,label in enumerate(label_list)
        }
        
        
        
        model_path=os.path.join(self.config.model_path,'config.json')

        config = json.load(open(model_path))
        config["id2label"] = id2label
        config["label2id"] = label2id
        json.dump(config, open(model_path,"w"))
        
                
        tokenizer = BertTokenizerFast.from_pretrained(self.config.tokenizer_path)
        # gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}
        model_fine_tuned = AutoModelForTokenClassification.from_pretrained(self.config.model_path)
        pipe = pipeline("ner", model=model_fine_tuned,tokenizer=tokenizer)

        
        print(text)

        output = pipe(text)
        
        print(output)

        return output


# p=PredictionPipeline()
# p.predict('bill gates is a person')