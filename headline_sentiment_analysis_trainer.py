"""Trainer for seaweed headline sentiment analysis."""
import pandas
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from huggingface_hub import login
from transformers import TrainingArguments, Trainer


with open('huggingface_tokens.txt', 'r', encoding='utf-8') as file:
    access_token_write = file.readline().strip()
    login(access_token_write, write_permission=True)

TOKENIZER = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased", token=access_token_write)
DATA_COLLATOR = DataCollatorWithPadding(tokenizer=TOKENIZER)


MODEL = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3)


def map_labels(row):
    label_dict = {-1: 0, 0: 1, 1: 2}
    return label_dict[row['sentiment']]


def compute_metrics(eval_pred):
    print(eval_pred)
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


def preprocess_function(examples):
    return TOKENIZER(examples["headline"], truncation=True)


def main():
    """Entry point."""

    df = pandas.read_csv('data/papers/froelich_headlines.csv')
    df['labels'] = df.apply(map_labels, axis=1)
    headline_dataset = Dataset.from_pandas(df)
    dataset = headline_dataset.train_test_split(test_size=0.2)
    print(dataset)
    tokenized_train = dataset['train'].map(preprocess_function, batched=True)
    tokenized_test = dataset['test'].map(preprocess_function, batched=True)

    repo_name = "wwf-seaweed-headline-sentiment"

    training_args = TrainingArguments(
       output_dir=repo_name,
       learning_rate=2e-5,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=16,
       num_train_epochs=2,
       weight_decay=0.01,
       save_strategy="epoch",
       push_to_hub=True
    )

    trainer = Trainer(
       model=MODEL,
       args=training_args,
       train_dataset=tokenized_train,
       eval_dataset=tokenized_test,
       tokenizer=TOKENIZER,
       data_collator=DATA_COLLATOR,
       compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    main()
