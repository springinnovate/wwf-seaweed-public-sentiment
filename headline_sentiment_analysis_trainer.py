"""Trainer for seaweed headline sentiment analysis."""
import argparse
import collections
import os

from datasets import Dataset
from datasets import load_metric
from huggingface_hub import login
from sklearn.metrics import confusion_matrix
from transformers import Adafactor
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers.optimization import AdafactorSchedule
import numpy as np
import pandas
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt


with open('huggingface_tokens.txt', 'r', encoding='utf-8') as file:
    access_token_write = file.readline().strip()
    login(access_token_write, write_permission=True)

MODELS_TO_TEST = [
    #'bert-base-uncased',
    #'roberta-base',
    #'google/electra-base-generator',
    'microsoft/deberta-v3-base',
    #'albert-base-v2',
    ]


def plot_learning_curves(trainer):
    # Extract training and validation loss
    training_loss = [entry['loss'] for entry in trainer.state.log_history if 'loss' in entry]
    validation_loss = [entry['eval_loss'] for entry in trainer.state.log_history if 'eval_loss' in entry]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Plot training and validation loss
    ax.plot(training_loss, label='Training loss')
    ax.plot(validation_loss, label='Validation loss')

    # Add title and labels
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()


def map_labels(row):
    label_dict = {-1: 0, 0: 1, 1: 2}
    return label_dict[row['sentiment']]

def map_label_to_word(label):
    if label == 0:
        return 'NEGATIVE'
    elif label == 1:
        return 'NEUTRAL'
    elif label == 2:
        return 'POSITIVE'

def compute_metrics(eval_pred):
    print(eval_pred)
    load_accuracy = load_metric("accuracy", trust_remote_code=True)
    load_f1 = load_metric("f1", trust_remote_code=True)

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average='weighted')["f1"]
    return {"accuracy": accuracy, "f1": f1}


def _make_preprocess_function(tokenizer):
    def _preprocess_function(examples):
        #return tokenizer(examples["headline"], truncation=True)
        return tokenizer(examples["headline"], truncation=True, padding='max_length', max_length=50)
    return _preprocess_function


def test_model(dataset, checkpoint_path_list):
    # Replace this with the actual path to your saved model checkpoint
    for checkpoint_path in checkpoint_path_list:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        tokenized_dataset = dataset.map(_make_preprocess_function(tokenizer), batched=True)

        # Convert to PyTorch tensors and create a DataLoader
        tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        dataloader = DataLoader(tokenized_dataset, batch_size=32)

        model.eval()  # Set the model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        predictions = []
        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                print(f'working on batch {index}')
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                break

        with open(f'{os.path.splitext(os.path.basename(checkpoint_path))[0]}_results.csv', 'w') as table:
            table.write('headline,sentiment,modeled sentiment\n')
            confusion_matrix = collections.defaultdict(lambda: collections.defaultdict(int))
            for headline, expected_id, actual_id in zip(
                    dataset['headline'], dataset['labels'], predictions):
                expected_label = map_label_to_word(expected_id)
                actual_label = map_label_to_word(actual_id)
                confusion_matrix[expected_label][actual_label] += 1
                table.write(
                    f'"{headline}",'
                    f'{expected_label},'
                    f'{actual_label}\n')
            table.write('\n')
            table.write(',' + ','.join(confusion_matrix) + '\n')
            for label in confusion_matrix:
                table.write(f'{label},' + ','.join(str(confusion_matrix[label][l]) for l in confusion_matrix))
                total_sum = sum(confusion_matrix[label].values())
                table.write(f'{confusion_matrix[label][label]/total_sum*100:.2f}%\n')

        print(f'{checkpoint_path} done')


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='WWF Seaweed Headline Sentiment Analysis')
    parser.add_argument(
        '--test_only', action='store_true',
        help='Ignore offshore mangrove and saltmarsh')
    parser.add_argument(
        '--model_checkpoint_paths', nargs='+', help='If test only, which model to test.')
    args = parser.parse_args()

    df = pandas.read_csv('data/papers/froelich_headlines.csv')
    df['labels'] = df.apply(map_labels, axis=1)
    headline_dataset = Dataset.from_pandas(df)
    if args.test_only:
        test_model(headline_dataset, args.model_checkpoint_paths)
        return

    dataset = headline_dataset.train_test_split(test_size=0.2)
    print(dataset)

    repo_name = "wwf-seaweed-headline-sentiment"

    training_args = TrainingArguments(
       output_dir=repo_name,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=16,
       num_train_epochs=1,
       weight_decay=0.01,
       save_strategy="epoch",
       push_to_hub=False
    )

    model_performance = open('modelperform.csv', 'w')
    for model_id in MODELS_TO_TEST:
        print(f'TRAINING ON: {model_id}')
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=access_token_write)

        tokenized_train = dataset['train'].map(
            _make_preprocess_function(tokenizer), batched=True)
        tokenized_test = dataset['test'].map(
            _make_preprocess_function(tokenizer), batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=3)
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None
        )
        lr_scheduler = AdafactorSchedule(optimizer)

        model_performance.write(f'\n{model_id}\n')
        model_performance.write('eval_loss,eval_accuracy\n')
        trainer = Trainer(
           model=model,
           args=training_args,
           train_dataset=tokenized_train,
           eval_dataset=tokenized_test,
           tokenizer=tokenizer,
           data_collator=data_collator,
           compute_metrics=compute_metrics,
           optimizers=(optimizer, lr_scheduler)
        )
        lowest_loss = None
        increase_loss_count = 0
        for epoch in range(1000):
            trainer.train()
            eval_results = trainer.evaluate()
            model_performance.write(
                f"{eval_results['eval_loss']},{eval_results['eval_accuracy']}\n")
            model_performance.flush()
            print(f'epoch {epoch+1}\n{eval_results}')
            if lowest_loss is not None and (lowest_loss < eval_results['eval_loss']):
                increase_loss_count += 1
                if increase_loss_count > 5:
                    break
            else:
                increase_loss_count = 0
                lowest_loss = eval_results['eval_loss']
            lowest_loss = eval_results['eval_loss']
            trainer.save_model(f"{repo_name}/{model_id.replace('/','-')}_{epoch+1}")
    model_performance.close()
    return
    # Make predictions on the test dataset
    predictions = trainer.predict(tokenized_test)
    predicted_labels = np.argmax(predictions.predictions, axis=-1)

    # True labels
    true_labels = predictions.label_ids

    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # # Plot the confusion matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title('Confusion Matrix')
    # plt.show()


if __name__ == '__main__':
    main()
