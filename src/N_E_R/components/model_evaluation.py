from transformers import AutoModelForTokenClassification, AutoTokenizer, BertTokenizerFast
from datasets import load_dataset, load_from_disk
import torch
import csv
import pandas as pd
import ast
from tqdm import tqdm
from seqeval.metrics import classification_report
from N_E_R.entity import ModelEvaluationConfig
import os

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Split the dataset into smaller batches that we can process simultaneously.
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def tokenize_and_align_labels(self, examples, tokenizer, label_all_tokens=True):
        tokenized_inputs = tokenizer(examples['tokens'], truncation=True, padding=True, is_split_into_words=True, return_tensors="pt")
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def calculate_metric_on_test_ds(self, dataset, model, tokenizer, 
                                    batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu"):
        all_predictions = []
        all_true_labels = []

        for batch in tqdm(self.generate_batch_sized_chunks(dataset, batch_size)):
            batch_tokens = [example['tokens'] for example in batch]
            batch_labels = [example['ner_tags'] for example in batch]

            inputs = tokenizer(batch_tokens, truncation=True, padding=True, is_split_into_words=True, return_tensors="pt")
            aligned_inputs = self.tokenize_and_align_labels({'tokens': batch_tokens, 'ner_tags': batch_labels}, tokenizer)

            # Ensure inputs and labels are aligned correctly
            if inputs['input_ids'].shape[1] != len(aligned_inputs['labels'][0]):
                max_len = min(inputs['input_ids'].shape[1], len(aligned_inputs['labels'][0]))
                for key in inputs.keys():
                    inputs[key] = inputs[key][:, :max_len]
                aligned_inputs['labels'] = [label[:max_len] for label in aligned_inputs['labels']]

            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = torch.tensor(aligned_inputs['labels']).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            true_labels = labels.cpu().numpy()

            for pred, true in zip(predictions, true_labels):
                pred = [p for p, t in zip(pred, true) if t != -100]
                true = [t for t in true if t != -100]
                all_predictions.append(pred)
                all_true_labels.append(true)

        return all_predictions, all_true_labels

    def read_data(self, path):
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

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BertTokenizerFast.from_pretrained(self.config.tokenizer_path)
        
        # Check if the model path is valid
        model = AutoModelForTokenClassification.from_pretrained(self.config.model_path).to(device)

        # Loading data
        dataset = self.read_data(self.config.data_path)

        predictions, true_labels = self.calculate_metric_on_test_ds(dataset, model, tokenizer, batch_size=16)

        # Assuming label_list is predefined or can be inferred from the dataset
        label_list =['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']  # Update this list as per your dataset

        y_true = [[label_list[l] for l in label] for label in true_labels]
        y_pred = [[label_list[l] for l in label] for label in predictions]

        report = classification_report(y_true, y_pred)

        # Save report to a CSV file
        with open(self.config.metric_file_name, 'w') as f:
            f.write(report)
        print(report)

# Example usage:
# config = ModelEvaluationConfig(tokenizer_path='bert-base-uncased', model_path='path_to_model', data_path='path_to_data.csv', metric_file_name='metrics.csv')
# evaluator = ModelEvaluation(config)
# evaluator.evaluate()
