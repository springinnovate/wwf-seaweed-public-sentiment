import collections
import glob
import logging
import os
import shutil

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from database_model_definitions import Article
from database_model_definitions import USER_CLASSIFIED_BODY_OPTIONS
from database import SessionLocal, init_db
from sqlalchemy import func

from datasets import concatenate_datasets
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk
from datasets import load_metric
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
import keyboard
import numpy
import pandas


OPTIONS = {
    index: key for index, key in enumerate(USER_CLASSIFIED_BODY_OPTIONS)
}

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
logging.getLogger('sentence_transformers').setLevel(logging.WARN)
logging.getLogger('httpx').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)

# used to determine how big one label set can be before it overwhelms the other
TRAINING_SIZE_FACTOR = 1.2
ABSTRACTS_COL = 'Abstract'
LABELS_COL = 'labels'
TRAIN_LABEL = 'train'
VALIDATION_LABEL = 'validation'
HOLDBACK_LABEL = 'holdback'
ABSTRACTS_DIR = 'data/scopus_2024_05_28'
ROOT_DIR = 'abstract_classifier_llm'
MODEL_PATH = os.path.join(ROOT_DIR, 'model')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TRAINING_SET_PATH = os.path.join(DATA_DIR, 'train')
VALIDATION_SET_PATH = os.path.join(DATA_DIR, 'validation')
HOLDOUT_SET_PATH = os.path.join(DATA_DIR, 'holdout')
DATASET_PATH = os.path.join(DATA_DIR, 'dataset')
LOGGING_DIR = os.path.join(ROOT_DIR, 'logging')

ACCURACY_METRIC = load_metric("accuracy")
PRECISION_METRIC = load_metric("precision")
RECALL_METRIC = load_metric("recall")
F1_METRIC = load_metric("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = numpy.argmax(logits, axis=-1)

    accuracy = ACCURACY_METRIC.compute(predictions=predictions, references=labels)
    precision = PRECISION_METRIC.compute(predictions=predictions, references=labels, average='weighted')
    recall = RECALL_METRIC.compute(predictions=predictions, references=labels, average='weighted')
    f1 = F1_METRIC.compute(predictions=predictions, references=labels, average='weighted')

    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision['precision'],
        'recall': recall['recall'],
        'f1': f1['f1'],
    }


def load_abstracts(abstracts_dir):
    # List all CSV files in the directory
    csv_files = [
        file_path
        for file_path in glob.glob(os.path.join(abstracts_dir, '*.csv'))]

    # Load each file and select the "Abstracts" column
    dataframes = []
    for file_path in csv_files:
        LOGGER.info(f'loading {file_path}')
        df = pandas.read_csv(file_path, usecols=[ABSTRACTS_COL])
        dataframes.append(df)
        break

    combined_dataframe = pandas.concat(dataframes, ignore_index=True)
    combined_dataframe = combined_dataframe.drop_duplicates()
    return combined_dataframe[ABSTRACTS_COL]


def process_key(abstract_series, abstract_list):
    abstract = None
    abstract_index = None

    def _print_choices():
        nonlocal abstract
        nonlocal abstract_index
        print('\n' + f'{abstract_index}: {abstract}')
        print('choose:\n\t' + '\n\t'.join([f'{key}: {classification_id}' for key, classification_id in LABEL_TO_DESC.items()]))
        print('press ESC to quit')

    def _process_key(event):
        nonlocal abstract
        nonlocal abstract_index
        choice = event.name
        if choice in LABELS:
            label = LABELS[choice]
            abstract_list[label].append(abstract)
            sample = abstract_series.sample(
                n=1, random_state=numpy.random.randint(numpy.iinfo(int).max))
            abstract = sample.iloc[0]
            abstract_index = sample.index[0]
            os.system('cls')
        elif choice == 'esc':
            print('detected escape, quitting')
            return 'q'
        else:
            print(f'!ERROR, unknown choice "{choice}" try again!\n')
        _print_choices()
    sample = abstract_series.sample(
        n=1, random_state=numpy.random.randint(numpy.iinfo(int).max))

    abstract = sample.iloc[0]
    abstract_index = sample.index[0]
    print('Starting training....')
    _print_choices()
    return _process_key



def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # Return None if value is not found in the dictionary


def print_options(options, selected):
    print("Options:")
    for i, option in options.items():
        print(f"{i}: {option} {'(selected)' if i in selected else ''}")


def get_user_choice(options, existing_subject):
    session = PromptSession()
    bindings = KeyBindings()
    selected = set()
    if existing_subject is not None:
        selected = set([
            get_key_from_value(OPTIONS, subject.strip())
            for subject in existing_subject.split(';')])
    choice = None

    @bindings.add('q')
    def _(event):
        nonlocal choice
        choice = 'quit'
        event.app.exit(result='c')

    @bindings.add('b')
    def _(event):
        nonlocal choice
        choice = 'back'
        event.app.exit(result=choice)

    def key_handler(event):
        nonlocal choice
        choice = int(event.data)
        if 0 <= choice < len(options):
            if choice in selected:
                selected.remove(choice)
            else:
                selected.add(choice)
            print("\033[F\033[K"*(len(OPTIONS)+1), end='')
            print_options(options, selected)

    for i in range(len(OPTIONS)):  # Add bindings for keys 1-9
        bindings.add(f"{i}")(key_handler)

    while True:
        try:
            print_options(options, selected)
            print("Press the number keys to select options, 'b' to go back, or press Enter to confirm: ", end='', flush=True)
            session.prompt("", key_bindings=bindings, default='')
            if isinstance(choice, str):
                return choice
            return [OPTIONS[index] for index in selected]
        except KeyboardInterrupt:
            return None


def highlight_keywords(body_text, keywords):
    highlighted_text = ""
    for word in body_text.split():
        for keyword in keywords:
            if keyword.lower() in word.lower():
                # Apply bold and highlighted background color
                highlighted_text += f"\033[1;48;5;231m{word}\033[0m "
                break
        else:
            highlighted_text += f"{word} "
    return highlighted_text


def main():

    init_db()
    session = SessionLocal()

    while True:
        if not os.path.exists(MODEL_PATH):
            LOGGER.info('initializing untrained model')
            tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', num_labels=2)
        else:
            LOGGER.info(f'loading model from {MODEL_PATH}')
            tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
            model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

        if os.path.exists(DATASET_PATH):
            LOGGER.debug(f'loading dataset from {DATASET_PATH}')
            dataset = load_from_disk(DATASET_PATH)
            LOGGER.info(f'loaded dataset:\n{dataset}')

            def _tokenize_function(examples):
                return tokenizer(
                    examples[ABSTRACTS_COL],
                    padding="max_length", truncation=True)
            tokenized_datasets = dataset.map(_tokenize_function, batched=True)
            training_args = TrainingArguments(
                output_dir='./results',
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=8,
                num_train_epochs=3,
                weight_decay=0.01,
                logging_dir=LOGGING_DIR,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets[TRAIN_LABEL],
                eval_dataset=tokenized_datasets[VALIDATION_LABEL],
                compute_metrics=compute_metrics,
            )

            trainer.train()
            holdout_results = trainer.evaluate(tokenized_datasets[HOLDBACK_LABEL])
            LOGGER.info(holdout_results)
            save_model = False
            if save_model:
                model.save_pretrained(MODEL_PATH)
                tokenizer.save_pretrained(MODEL_PATH)
        else:
            dataset = DatasetDict({
                TRAIN_LABEL: Dataset.from_dict({ABSTRACTS_COL: [], LABELS_COL: []}),
                VALIDATION_LABEL: Dataset.from_dict({ABSTRACTS_COL: [], LABELS_COL: []}),
                HOLDBACK_LABEL: Dataset.from_dict({ABSTRACTS_COL: [], LABELS_COL: []}),
            })

        print('ready to train, press ENTER to start')
        label_to_abstract_list = collections.defaultdict(list)
        keyboard.on_press(process_key(abstract_series, label_to_abstract_list))
        result = keyboard.wait('esc')
        keyboard.unhook_all()

        unique_elements, counts = numpy.unique(
            dataset[TRAIN_LABEL][LABELS_COL], return_counts=True)
        unique_counts = collections.defaultdict(
            int, zip(unique_elements, counts))
        for label, abstract_list in label_to_abstract_list.items():
            train_size, test_size, holdback_size = split_list(
                len(abstract_list), [0.8, 0.1, 0.1])
            LOGGER.debug(f'{train_size} {test_size} {holdback_size}')
            # if the training size for this label exceeds the training size of the other label
            # by some large amount, kick the extra over to the holdback set
            overshoot_count = unique_counts[label] - TRAINING_SIZE_FACTOR*unique_counts[(label+1)%2]
            if overshoot_count > 0:
                correcting_count = int(min(overshoot_count, train_size))
                train_size -= correcting_count
                holdback_size += correcting_count

            start_index = 0
            new_training_data = {}
            for dataset_type, dataset_size in [
                    (TRAIN_LABEL, train_size),
                    (VALIDATION_LABEL, test_size),
                    (HOLDBACK_LABEL, holdback_size)]:
                LOGGER.debug(f'{dataset_type} {start_index}:{start_index}+{dataset_size} ')
                new_training_data = Dataset.from_dict({
                    ABSTRACTS_COL: abstract_list[
                        start_index:start_index + dataset_size],
                    LABELS_COL: [label] * dataset_size
                })
                start_index += dataset_size
                LOGGER.debug(f'{dataset}')
                LOGGER.debug(f'{new_training_data}')
                dataset[dataset_type] = concatenate_datasets(
                    [dataset[dataset_type], new_training_data])
                LOGGER.debug(f'after: {dataset}')

        temp_path = DATASET_PATH + "_temp"
        dataset.save_to_disk(temp_path)
        shutil.rmtree(DATASET_PATH)
        shutil.move(temp_path, DATASET_PATH)
        print(result)
        print(f'classifications in {dataset}')
        del dataset


def split_list(n_elements, split_proportions):
    split_size = [max(1, int(n_elements * prop)) for prop in split_proportions]
    overshoot_count = sum(split_size) - n_elements
    while overshoot_count > 0:
        argmax = numpy.argmax(split_size)
        argmax_val = split_size[argmax]
        if argmax_val <= 1:
            raise ValueError('not enough elements to evenly distribute')
        argsecondmax_val = numpy.partition(split_size, -2)[-2]
        if argsecondmax_val < 1:
            raise ValueError('not enough elements to evenly distribute')
        dec_amount = min(overshoot_count, split_size[argmax] - argsecondmax_val)
        split_size[argmax] -= dec_amount
        overshoot_count -= dec_amount
    return split_size


if __name__ == '__main__':
    main()
