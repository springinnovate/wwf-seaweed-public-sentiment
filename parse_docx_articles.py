"""Tracer code to figure out how to parse out DocX files."""
import argparse
import glob
import re
import time

import pandas
from docx import Document
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

MODEL_PATH = 'wwf-seaweed-headline-sentiment/microsoft-deberta-v3-base_6'
HEADLINE_LABEL_TO_SENTIMENT = {
    'LABEL_0': 'bad',
    'LABEL_1': 'neutral',
    'LABEL_2': 'good',
}


def parse_docx(file_path, headline_sentiment_model):
    start_time = time.time()
    print(f'parsing {file_path}')
    class Parser:
        def __init__(self):
            doc = Document(file_path)
            self.iter = iter(doc.paragraphs)

        def __iter__(self):
            return self

        def __next__(self):
            while True:
                val = next(self.iter).text.strip()
                if val != '':
                    return val

    paragaph_iter = Parser()
    while True:
        text = next(paragaph_iter)
        if text != '':
            break
    headline_text = text
    publication_text = next(paragaph_iter)
    date_text = next(paragaph_iter)
    for text in paragaph_iter:
        if text == 'Body':
            break

    body_text = ''
    for text in paragaph_iter:
        if text == 'Classification' or text.startswith('----------------'):
            break
        body_text += text

    result = {
        'subject': '',
        'industry': '',
        'geographic': '',
        'load-date': '',
        'headline': headline_text,
        'body': body_text,
        'publication': publication_text,
        'date': date_text,
    }

    try:
        result['year'] = re.search(r'\d{4}', date_text).group()
    except AttributeError:
        result['year'] = 'unknown'

    for text in paragaph_iter:
        tag = text.split(':')[0]
        if tag in result:
            result[tag] = text

    headline_sentiment = headline_sentiment_model([headline_text])[0]
    result['headline-sentiment'] = HEADLINE_LABEL_TO_SENTIMENT[headline_sentiment['label']]
    result['headline-sentiment-score'] = headline_sentiment['score']

    print(f'time to parse = {time.time()-start_time}s')

    return result


def main():
    parser = argparse.ArgumentParser(description='parse docx')
    parser.add_argument('path_to_files', help='Path/wildcard to docx files')
    args = parser.parse_args()

    print(f'load {MODEL_PATH}')
    model = pipeline(
        'sentiment-analysis', model=MODEL_PATH, device='cuda')
    print('loaded...')

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_list = []
        for index, file_path in enumerate(glob.glob(args.path_to_files)):
            future = executor.submit(parse_docx, file_path, model)
            future_list.append(future)
            if index > 4:
                break
        df = pandas.DataFrame.from_records([future.result() for future in future_list])
        df.to_csv('_parse_doc.csv')


if __name__ == '__main__':
    main()
