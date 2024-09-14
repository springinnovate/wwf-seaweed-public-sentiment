"""Tracer code to figure out how to parse out DocX files."""
import sys
from pathlib import Path
import argparse
import glob
import time
import re

import pandas
from striprtf.striprtf import rtf_to_text
from concurrent.futures import as_completed
from docx import Document
from concurrent.futures import ProcessPoolExecutor
from database_model_definitions import Article
from database_operations import upsert_articles
from database import SessionLocal, init_db


def parse_as_bibstyle(orignal_file_path, paragraph_iter):
    print('parsing bibstyle')
    start_time = time.time()
    end_of_document = False
    while True:
        text = next(paragraph_iter)
        if text == 'End of Document':
            end_of_document = True
        elif end_of_document and text != 'Bibliography':
            headline_text = text
            break
        else:
            end_of_document = False
    headline_text = text

    article_list = []
    while True:  # parse all the articles in the docx
        try:
            publication_text = next(paragraph_iter)
            date_text = next(paragraph_iter)
            for text in paragraph_iter:
                if text == 'Body':
                    break

            body_text = ''
            for text in paragraph_iter:
                if text == 'End of Document':
                    break
                body_text += text

            new_article = Article(
                headline=headline_text,
                body=body_text,
                date=date_text,
                publication=publication_text,
                source_file=str(orignal_file_path)
            )
            article_list.append(new_article)
            headline_text = next(paragraph_iter)
        except StopIteration:
            break

    print(f'time to parse = {time.time()-start_time:.2f}s')
    return article_list


def parse_as_user_name(original_file_path, paragraph_iter):
    start_time = time.time()
    seen_number = False
    while True:
        text = next(paragraph_iter)
        isdigit = text[0].isdigit()
        if seen_number and not isdigit:
            headline_text = text
            break
        if not seen_number and isdigit:
            seen_number = True
    headline_text = text
    article_list = []
    while True:
        try:
            publication_text = next(paragraph_iter)
            date_text = next(paragraph_iter)
            for text in paragraph_iter:
                if text == 'Body':
                    break

            body_text = ''
            for text in paragraph_iter:
                if text == 'End of Document':
                    break
                body_text += text

            new_article = Article(
                headline=headline_text,
                body=body_text,
                date=date_text,
                publication=publication_text,
                source_file=str(original_file_path)
            )
            article_list.append(new_article)
            headline_text = next(paragraph_iter)
        except StopIteration:
            break
    print(f'time to parse = {time.time()-start_time}s')
    return article_list


def parse_as_old_rtf(file_path, paragraph_iter):
    start_time = time.time()
    seen_body = False
    headline = ''
    publication = ''
    date = ''
    body_text = ''
    article_list = []
    while True:
        try:
            line = next(paragraph_iter)
            if line == 'End of Document':
                if seen_body:
                    seen_body = False
                    new_article = Article(
                        headline=headline,
                        body=body_text,
                        date=date,
                        publication=publication,
                        source_file=str(file_path),
                    )
                    article_list.append(new_article)
                else:
                    headline = next(paragraph_iter)
                    publication = next(paragraph_iter)
                    date = next(paragraph_iter)
            else:
                body_text += line
            if line == 'Body':
                seen_body = True
                body_text = ''
        except StopIteration:
            break
    print(f'time to parse = {time.time()-start_time}s')
    return article_list


def parse_docx(file_path):
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

    paragraph_iter = Parser()
    first_line = next(paragraph_iter)
    if first_line.startswith('Bibliography'):
        return parse_as_bibstyle(file_path, paragraph_iter)
    elif first_line.startswith('User Name: ='):
        return parse_as_user_name(file_path, paragraph_iter)
    else:
        return parse_as_old_rtf(file_path, paragraph_iter)


def parse_csv(csv_path):
    # Group by item_title, item_pub_date, and item_url
    df = pandas.read_csv(csv_path)
    possible_item_titles = ['item_title', 'item_title.x']
    for item_title in possible_item_titles:
        if item_title in df.columns:
            break
    grouped = df.groupby([item_title, 'item_pub_date', 'item_url'])

    # Concatenate the 'item_text's into 'body', ignoring NaNs and empty strings
    df_combined = grouped['item_text'].apply(
        lambda x: ' '.join(x.dropna().astype(str).str.strip())
    ).reset_index(name='body')

    article_list = []
    for index, row in df_combined.iterrows():
        new_article = Article(
            headline=row[item_title],
            body=row['item_text'],
            date=row['item_pub_date'],
            source_file=str(csv_path),
        )
        article_list.append(new_article)
    return article_list


def main():
    parser = argparse.ArgumentParser(description='parse docx')
    #parser.add_argument('path_to_files', nargs='+', help='Path/wildcard to docx files')
    args = parser.parse_args()

    path_to_files = [
        "data/papers/Aquaculture-Regional-2024/*.docx",
        "data/papers/2024-Seaweed-AccessWorldNews/*.docx",
        "data/papers/2024-Seaweed-NexisUni/*.docx",
        "data/papers/2024-Seaweed-Regional/*.docx",
        "data/papers/Aquaculture-AccessWorldNews-2024/*.docx",
        "data/papers/Aquaculture-NexisUni-2024/*.docx",
        "data/papers/Aquaculture-Regional-2024/*.docx",
        "data/papers/2024-Seaweed-AccessWorldNews/*.csv",
        "data/papers/2024-Seaweed-NexisUni/*.csv",
        "data/papers/2024-Seaweed-Regional/*.csv",
        "data/papers/Aquaculture-AccessWorldNews-2024/*.csv",
        "data/papers/Aquaculture-NexisUni-2024/*.csv",
    ]

    init_db()
    db = SessionLocal()
    doc_path_list = [
        pdf_file_path
        for pdf_file_glob in path_to_files  # args.path_to_files
        for pdf_file_path in Path().rglob(pdf_file_glob)]

    article_list = []
    for doc_path in doc_path_list:
        if str(doc_path).lower().endswith('.docx'):
            article_list.extend(parse_docx(doc_path))
        elif str(doc_path).lower().endswith('.csv'):
            article_list.extend(parse_csv(doc_path))
        else:
            raise RuntimeError(f'unknown file {doc_path}')
    print(f'upserting {len(article_list)} articles')
    upsert_articles(db, article_list)
    db.commit()
    db.close()

    return


    with ProcessPoolExecutor() as executor:
        future_list = []
        for index, file_path in enumerate(doc_path_list):
            future = executor.submit(parse_docx, file_path)
            future_list.append(future)
        article_list = []
        for future in as_completed(future_list):
            article_list.extend(future.result())
        article_list = [future.result() for future in future_list]
    print(f'upserting {len(article_list)} articles')
    upsert_articles(db, article_list)

    db.commit()
    db.close()


if __name__ == '__main__':
    main()
