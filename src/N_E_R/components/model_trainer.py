import os
import sys
import csv
import ast
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, Trainer, BertTokenizerFast, DataCollatorForTokenClassification, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from N_E_R.entity import ModelTrainerConfig
import torch
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def read_data(self,path):
        dataset = []
        with open(path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                row["input_ids"] = ast.literal_eval(row["input_ids"])
                row["attention_mask"] = ast.literal_eval(row["attention_mask"])
                row["labels"] = ast.literal_eval(row["labels"])
                row["token_type_ids"] = ast.literal_eval(row["token_type_ids"])
                row["tokens"] = ast.literal_eval(row["tokens"])
                row["ner_tags"] = ast.literal_eval(row["ner_tags"])
                dataset.append(row)
        return dataset

    def train(self):
        config=self.config
        transformed_train_data_set = self.read_data(config.train_data_path)
        transformed_train_data_set= Dataset.from_pandas(pd.DataFrame(transformed_train_data_set))
        transformed_test_data_set = self.read_data(config.test_data_path)
        transformed_test_data_set= Dataset.from_pandas(pd.DataFrame(transformed_test_data_set))
        transformed_val_data_set = self.read_data(config.val_data_path)
        transformed_val_data_set= Dataset.from_pandas(pd.DataFrame(transformed_val_data_set))
        


        # Find the number of unique labels
        # unique_labels = set(label for example in data for label in example["labels"])
        # num_labels = len(unique_labels)

        # Define training arguments
        args = TrainingArguments(
            "test-ner",
            evaluation_strategy=config.evaluation_strategy,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            num_train_epochs=config.num_train_epochs,
            weight_decay=config.weight_decay,
        )

        # Define model, tokenizer, and data_collator here
        model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=config.num_labels)
        tokenizer = BertTokenizerFast.from_pretrained(r"C:\Users\balkr\Name_Entity_Recognition\tokenizer")
        data_collator = DataCollatorForTokenClassification(tokenizer)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=transformed_test_data_set,
            eval_dataset=transformed_val_data_set,  # You may want to use a separate eval dataset
            data_collator=data_collator,
            tokenizer=tokenizer,
            # Define compute_metrics function if you want to evaluate metrics
        )
        trainer.train()
        model.save_pretrained(config.root_dir)

# # Example usage
# m = ModelTrainer()
# m.model()
# C:\Users\balkr\Name_Entity_Recognition\Entity_Recognition\components\model_trainer.py

# model.save_pretrained("ner_model")

