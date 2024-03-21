"""Trainer for seaweed subject analysis."""
from database_model_definitions import Article
from database import SessionLocal

import logging
import sys
import collections
import os

from database_model_definitions import RELEVANT_SUBJECT_TO_LABEL
from database_model_definitions import AQUACULTURE_SUBJECT_TO_LABEL
from datasets import Dataset
from datasets import load_metric
from huggingface_hub import login
from torch.utils.data import DataLoader
from transformers import Adafactor
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers.optimization import AdafactorSchedule
import numpy as np
import pandas
import torch

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
LOGGER.setLevel(logging.DEBUG)
logging.getLogger('ecoshard.fetch_data').setLevel(logging.INFO)


with open('huggingface_tokens.txt', 'r', encoding='utf-8') as file:
    access_token_write = file.readline().strip()
    login(access_token_write, write_permission=True)

MODEL_ID = 'allenai/longformer-base-4096'
DATA_KEY = 'body'
LABEL_KEY = 'subject'



def map_labels(subject_to_label_dict, key):
    def _map_labels(row):
        # give me the first hit by priority
        for subject, label in sorted(subject_to_label_dict.items()):
            if subject.lower() in row[key].lower():
                return label
        return None
    return _map_labels


def map_label_to_word(label):
    if label == 0:
        return 'NEGATIVE'
    elif label == 1:
        return 'NEUTRAL'
    elif label == 2:
        return 'POSITIVE'


def compute_metrics(eval_pred):
    print(f'eval_pred: {eval_pred}')
    load_accuracy = load_metric("accuracy", trust_remote_code=True)
    load_f1 = load_metric("f1", trust_remote_code=True)

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    LOGGER.debug(f'************* in compute metrics: logits: {logits}, labels: {labels}')
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average='weighted')["f1"]
    return {"accuracy": accuracy, "f1": f1}


def _make_preprocess_function(tokenizer):
    def _preprocess_function(examples):
        return tokenizer(examples[DATA_KEY], truncation=True, padding='max_length', max_length=50)
    return _preprocess_function


def test_model(dataset, checkpoint_path_list):
    # Replace this with the actual path to your saved model checkpoint
    for checkpoint_path in checkpoint_path_list:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        tokenized_dataset = dataset.map(_make_preprocess_function(tokenizer), batched=True)
        print(len(tokenized_dataset))
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
        print(len(dataset[DATA_KEY]))
        print(len(dataset['labels']))
        print(len(predictions))

        with open(f'{os.path.splitext(os.path.basename(checkpoint_path))[0]}_results.csv', 'w') as table:
            table.write(f'{DATA_KEY},{LABEL_KEY},modeled {LABEL_KEY}\n')
            confusion_matrix = collections.defaultdict(lambda: collections.defaultdict(int))
            for headline, expected_id, actual_id in zip(
                    dataset[DATA_KEY], dataset['labels'], predictions):
                expected_label = map_label_to_word(expected_id)
                actual_label = map_label_to_word(actual_id)
                confusion_matrix[expected_label][actual_label] += 1
                headline = headline.replace('"', '')
                table.write(
                    f'"{headline}",'
                    f'{expected_label},'
                    f'{actual_label}\n')
            table.write('\n')
            table.write(',' + ','.join(confusion_matrix) + ',accuracy\n')
            for label in confusion_matrix:
                table.write(f'{label},' + ','.join(str(confusion_matrix[label][l]) for l in confusion_matrix))
                total_sum = sum(confusion_matrix[label].values())
                table.write(f',{confusion_matrix[label][label]/total_sum*100:.2f}%\n')

        print(f'{checkpoint_path} done')


RELEVANT_SUBJECT_TAGS = {
    'OTHER AQUACULTURE',
    'SEAWEED AQUACULTURE'
}

IRRELEVANT_SUBJECT_TAGS = {
    'NOT NEWS',
    'NOT ENGLISH',
    'NOT AQUACULTURE',
}

SEAWEED_TAG = 'SEAWEED AQUACULTURE'
AQUACULTURE_TAG = 'OTHER AQUACULTURE'

def main():
    """Entry point."""
    """
    Column is user_classified_body_subject, the values are:

    We want two phase

    relevant
        * OTHER AQUACULTURE
        * SEAWEED AQUACULTURE
    irrelevant
        * NOT NEWS
        * NOT ENGLISH
        * NOT AQUACULTURE

    Then subject:
        * OTHER AQUACULTURE
        * SEAWEED AQUACULTURE

    Train relevant/irrelevant model
    Train seaweed/other model
    """
    session = SessionLocal()
    subjects_bodies = [
        (article.user_classified_body_subject, article.body) for article in
        session.query(Article).filter(
            Article.body != None,
            Article.user_classified_body_subject != None,
            Article.user_classified_body_subject != '')
        .all()]

    for classification_phase, subject_to_label in [
            ('relevant-irrelevant', RELEVANT_SUBJECT_TO_LABEL),
            ('aquaculture-type', AQUACULTURE_SUBJECT_TO_LABEL),
            ]:
        df = pandas.DataFrame(subjects_bodies, columns=[LABEL_KEY, DATA_KEY])
        df['labels'] = df.apply(
            map_labels(subject_to_label, LABEL_KEY), axis=1)
        df = df.dropna(subset=['labels'])

        df.to_csv(f'{classification_phase}_out.csv')
        body_dataset = Dataset.from_pandas(df)
        print(f'{classification_phase}: {body_dataset}')
        continue
        dataset = body_dataset.train_test_split(test_size=0.2)
        LOGGER.debug(f'this is how the dataset is broken down: {dataset}')
        repo_name = f"wwf-seaweed-body-subject-{classification_phase}"
        training_args = TrainingArguments(
           output_dir=repo_name,
           per_device_train_batch_size=1,
           per_device_eval_batch_size=1,
           num_train_epochs=1,
           weight_decay=0.01,
           save_strategy="epoch",
           push_to_hub=False,
           gradient_accumulation_steps=32,
        )
        model_performance = open(
            f'modelperform-{classification_phase}.csv', 'w')

        print(f'TRAINING ON: {MODEL_ID} / {classification_phase}')
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, token=access_token_write)
        tokenized_train = dataset['train'].map(
            _make_preprocess_function(tokenizer), batched=True)
        tokenized_test = dataset['test'].map(
            _make_preprocess_function(tokenizer), batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        num_labels = max(subject_to_label.values())+1
        print(f'NMODELS: {num_labels}')
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=num_labels)
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None
        )
        lr_scheduler = AdafactorSchedule(optimizer)

        model_performance.write(f'\n{MODEL_ID}-{classification_phase}\n')
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
            trainer.save_model(f"{repo_name}/{MODEL_ID.replace('/','-')}_{epoch+1}")
        model_performance.close()


if __name__ == '__main__':
    main()
