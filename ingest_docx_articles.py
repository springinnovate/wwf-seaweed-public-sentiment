"""Tracer code to figure out how to parse out DocX files."""
import argparse
import glob
import re
import time

import pandas
from docx import Document
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session
from database_model_definitions import Article, AIResultHeadline
from datbase_operations import upsert_articles
from database import SessionLocal, init_db

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
        result['year'] = int(re.search(r'\d{4}', date_text).group())
    except AttributeError:
        result['year'] = None

    for text in paragaph_iter:
        tag = text.split(':')[0].lower()
        if tag in result:
            result[tag] = '.'.join(text.split(':')[1:]).replace(
                '\xa0', ' ').strip()

    headline_sentiment_result = headline_sentiment_model([headline_text])[0]

    headline_sentiment = AIResultHeadline(
        value=HEADLINE_LABEL_TO_SENTIMENT[headline_sentiment_result['label']],
        score=headline_sentiment_result['score'])

    print(f'time to parse = {time.time()-start_time}s')
    print(result)

    new_article = Article(
        headline=headline_text,
        body=body_text,
        date=date_text,
        year=result['year'],
        publication=publication_text,
        headline_sentiment_ai=headline_sentiment,
        )
    return new_article


def main():
    parser = argparse.ArgumentParser(description='parse docx')
    parser.add_argument('path_to_files', help='Path/wildcard to docx files')
    args = parser.parse_args()

    init_db()
    db = SessionLocal()

    print(f'load {MODEL_PATH}')
    model = pipeline(
        'sentiment-analysis', model=MODEL_PATH, device='cuda')
    print('loaded...')

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_list = []
        for index, file_path in enumerate(glob.glob(args.path_to_files)):
            future = executor.submit(parse_docx, file_path, model)
            future_list.append(future)
        db.add_all([future.result() for future in future_list])

    db.commit()
    db.close()


if __name__ == '__main__':
    main()
