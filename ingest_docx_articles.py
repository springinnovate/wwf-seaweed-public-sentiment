"""Tracer code to figure out how to parse out DocX files."""
import argparse
import glob
import time
import re

from docx import Document
from concurrent.futures import ProcessPoolExecutor

from database_model_definitions import Article
from database_operations import upsert_articles
from database import SessionLocal, init_db


def parse_docx(file_path):
    start_time = time.time()
    #print(f'parsing {file_path}')

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

    relevant_subjects = []
    locations = []
    for line in paragaph_iter:
        if line.startswith('Subject:\xa0'):
            subjects = re.findall(r'([A-Z][A-Z&\s]+)\s\(\d+%\)', line)
            relevant_subjects = [subject for subject in subjects]
        if line.startswith('Geographic'):
            locations = re.findall(r'([A-Z][A-Z&\s,]+)\s\(\d+%\)', line)
    relevant_subject_str = ', '.join(relevant_subjects)
    location_str = '; '.join(locations)
    print(f'time to parse = {time.time()-start_time}s')
    new_article = Article(
        headline=headline_text,
        body=body_text,
        date=date_text,
        publication=publication_text,
        source_file=file_path,
        ground_truth_body_subject=relevant_subject_str,
        ground_truth_body_location=location_str,
        )
    return new_article


def main():
    parser = argparse.ArgumentParser(description='parse docx')
    parser.add_argument('path_to_files', help='Path/wildcard to docx files')
    args = parser.parse_args()

    init_db()
    db = SessionLocal()

    with ProcessPoolExecutor() as executor:
        future_list = []
        for index, file_path in enumerate(glob.glob(args.path_to_files)):
            future = executor.submit(parse_docx, file_path)
            future_list.append(future)
        article_list = [future.result() for future in future_list]
    print(f'upserting {len(article_list)} articles')
    upsert_articles(db, article_list)

    db.commit()
    db.close()


if __name__ == '__main__':
    main()
