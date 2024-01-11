"""Trainer for seaweed headline sentiment analysis."""
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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
from transformers import Adafactor
from transformers.optimization import AdafactorSchedule


with open('huggingface_tokens.txt', 'r', encoding='utf-8') as file:
    access_token_write = file.readline().strip()
    login(access_token_write, write_permission=True)

MODEL_ID = 'bert-base-uncased'  # 0.854 after 5 epochs then loss starts to increase agasin
#MODEL_ID = 'roberta-base'  # 0.821 after 4 epochs then loss incrases
#MODEL_ID = 'google/electra-base-generator' # got up to 0.806 beffore loss increases again
#MODEL_ID = 'microsoft/deberta-v3-base'  # .865 after 6 epoch
#MODEL_ID = 'albert-base-v2' # 0.844 after 6 with no significant further improvement
TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_ID, token=access_token_write)
DATA_COLLATOR = DataCollatorWithPadding(tokenizer=TOKENIZER)

MODEL = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=3)

MAX_EPOCHS = 1000

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


def preprocess_function(examples):
    return TOKENIZER(examples["headline"], truncation=True)


def compute_metrics(eval_pred):
    print(eval_pred)
    load_accuracy = load_metric("accuracy", trust_remote_code=True)
    load_f1 = load_metric("f1", trust_remote_code=True)

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average='weighted')["f1"]
    return {"accuracy": accuracy, "f1": f1}


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

    optimizer = Adafactor(
        MODEL.parameters(),
        scale_parameter=True,
        relative_step=True,
        warmup_init=True,
        lr=None  # This can be set to a specific learning rate or left as None
    )
    lr_scheduler = AdafactorSchedule(optimizer)
    training_args = TrainingArguments(
       output_dir=repo_name,
       #learning_rate=2e-5,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=16,
       num_train_epochs=1,
       weight_decay=0.01,
       save_strategy="epoch",
       push_to_hub=False
    )

    model_performance = open('modelperform.csv', 'w')
    for model_id in [
            'bert-base-uncased', 'roberta-base',
            'google/electra-base-generator', 'microsoft/deberta-v3-base',
            'albert-base-v2']:
        model_performance.write(f'\n{model_id}\n')
        model_performance.write('eval_loss,eval_accuracy\n')
        trainer = Trainer(
           model=model_id,
           args=training_args,
           train_dataset=tokenized_train,
           eval_dataset=tokenized_test,
           tokenizer=TOKENIZER,
           data_collator=DATA_COLLATOR,
           compute_metrics=compute_metrics,
           optimizers=(optimizer, lr_scheduler)
        )
        last_loss = None
        for epoch in range(MAX_EPOCHS):
            trainer.train()
            eval_results = trainer.evaluate()
            model_performance.write(
                f"{eval_results['eval_loss'],eval_results['eval_accuracy']}\n")
            model_performance.flush()
            print(eval_results)
            if last_loss is not None and (last_loss < eval_results['eval_loss']):
                break
            last_loss = eval_results['eval_loss']
            trainer.save_model(f"{repo_name}/model_epoch_{epoch+1}")
    model_performance.close()
    return
    # Make predictions on the test dataset
    predictions = trainer.predict(tokenized_test)
    predicted_labels = np.argmax(predictions.predictions, axis=-1)

    # True labels
    true_labels = predictions.label_ids

    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    main()
