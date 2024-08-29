"""Trainer for seaweed subject analysis."""
from database_model_definitions import Article
from database import SessionLocal

import logging
import sys
import collections
import os

from database_model_definitions import RELEVANT_SUBJECT_TO_LABEL, IRRELEVANT_LABEL
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


def map_labels(subject_to_label_dict, key, reject_set):
    def _map_labels(row):
        # give me the first hit by priority
        for subject, label in sorted(subject_to_label_dict.items()):
            if subject.lower() in reject_set:
                return None
            if subject.lower() in row[key].lower():
                return label
        return None
    return _map_labels


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

    irrelevant_set = {
        subject for subject, label in RELEVANT_SUBJECT_TO_LABEL.items()
        if label == IRRELEVANT_LABEL
    }

    for classification_phase, subject_to_label, reject_set in [
            #('relevant-irrelevant', RELEVANT_SUBJECT_TO_LABEL, set()),
            ('aquaculture-type', AQUACULTURE_SUBJECT_TO_LABEL, irrelevant_set),
            ]:
        df = pandas.DataFrame(subjects_bodies, columns=[LABEL_KEY, DATA_KEY])
        df['labels'] = df.apply(
            map_labels(subject_to_label, LABEL_KEY, reject_set), axis=1)
        df = df.dropna(subset=['labels']).reset_index(drop=True)
        df['labels'] = df['labels'].astype(int)

        df.to_csv(f'{classification_phase}_out.csv')
        body_dataset = Dataset.from_pandas(df)
        print(f'{classification_phase}: {body_dataset}')
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
