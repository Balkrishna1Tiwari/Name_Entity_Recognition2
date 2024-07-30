import os
import csv
import pandas as pd
from transformers import BertTokenizerFast
from N_E_R.logging import logger
from N_E_R.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def read_data(self, file_path):
        dataset = []
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for idx, row in enumerate(reader):
                try:
                    row["tokens"] = row["tokens"].split()
                    row["ner_tags"] = list(map(int, row["ner_tags"].split()))
                    dataset.append(row)
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Error parsing line {idx + 1} in {file_path}: {e}")
                    logger.error(f"Row content: {row}")
        return dataset

    def tokenize_and_align_labels(self, example, label_all_tokens=True):
        tokenized_inputs = self.tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        word_ids = tokenized_inputs.word_ids()

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(int(example["ner_tags"][word_idx]))
            else:
                label_ids.append(int(example["ner_tags"][word_idx]) if label_all_tokens else -100)
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = label_ids
        tokenized_inputs["tokens"] = example["tokens"]
        tokenized_inputs["ner_tags"] = example["ner_tags"]
        return tokenized_inputs

    def process_and_save(self, split_name, file_path):
        data = self.read_data(file_path)
        transformed_data = [self.tokenize_and_align_labels(example) for example in data]

        os.makedirs(self.config.root_dir, exist_ok=True)

        transformed_file_path = os.path.join(self.config.root_dir, f"{split_name}_transformed.csv")

        # Convert the transformed data to a DataFrame
        transformed_df = pd.DataFrame({
            'input_ids': [x['input_ids'] for x in transformed_data],
            'attention_mask': [x['attention_mask'] for x in transformed_data],
            'token_type_ids': [x['token_type_ids'] for x in transformed_data],
            'labels': [x['labels'] for x in transformed_data],
            'tokens': [x['tokens'] for x in transformed_data],
            'ner_tags': [x['ner_tags'] for x in transformed_data]
        })

        # Save the transformed data to a CSV file
        transformed_df.to_csv(transformed_file_path, index=False)

        logger.info(f"Transformed {split_name} data saved to CSV file")

        return transformed_file_path

    def initiate_data_transformation(self):
        logger.info("Entered the initiate_data_transformation method of Data transformation class")

        train_path = self.process_and_save('train', self.config.data_path_train)
        test_path = self.process_and_save('test', self.config.data_path_test)
        validation_path = self.process_and_save('val', self.config.data_path_val)

        logger.info("Transformed data saved to CSV files for train, test, and validation splits")
        self.tokenizer.save_pretrained(self.config.tokenizer_path)
