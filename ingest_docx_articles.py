"""Tracer code to figure out how to parse out DocX files."""
import sys
from pathlib import Path
import argparse
import glob
import time
import re

import pandas
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
            print(f'NEW ARTICLE ON THIS TEXT: {text}')
            headline_text = text
            break
        else:
            end_of_document = False
    headline_text = text

    article_list = []
    while True:  # parse all the articles in the docx
        try:
            print(f'headline text: {headline_text}')
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
            print(f'BODY: {body_text[:1000]}')

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
        print(text)
        isdigit = text[0].isdigit()
        if seen_number and not isdigit:
            headline_text = text
            break
        if not seen_number and isdigit:
            seen_number = True
    headline_text = text
    print(f'headline text: {headline_text}')
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
            )
            article_list.append(new_article)
            headline_text = next(paragraph_iter)
        except StopIteration:
            break
    print(f'time to parse = {time.time()-start_time}s')
    return article_list


def parse_docx(file_path):
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

    paragraph_iter = Parser()
    first_line = next(paragraph_iter)
    if first_line.startswith('Bibliography'):
        return parse_as_bibstyle(file_path, paragraph_iter)
    elif first_line.startswith('User Name: ='):
        return parse_as_user_name(file_path, paragraph_iter)
    else:
        print(f'"{file_path}" has unknown first line: {first_line}')
        return


def parse_rtf(doc_path):
    pass


def parse_csv(csv_path):
    # Group by item_title, item_pub_date, and item_url
    df = pandas.read_csv(csv_path)
    possible_item_titles = ['item_title', 'item_title.x']
    for item_title in possible_item_titles:
        if item_title in df.columns:
            break
    print(df)
    grouped = df.groupby([item_title, 'item_pub_date', 'item_url'])
    print(grouped)

    # Concatenate the 'item_text's into 'body', ignoring NaNs and empty strings
    df_combined = grouped['item_text'].apply(
        lambda x: ' '.join(x.dropna().astype(str).str.strip())
    ).reset_index(name='body')

    # Display the combined DataFrame
    print("\nCombined DataFrame:")
    print(df_combined)
    return


def main():
    parser = argparse.ArgumentParser(description='parse docx')
    parser.add_argument('path_to_files', nargs='+', help='Path/wildcard to docx files')
    args = parser.parse_args()

    parse_csv('./data/papers/2024-Seaweed-Regional/seaweedRegionalSearch_Manual_ScrapedData2024.csv')
    #parse_csv('./data/papers/2024-Seaweed-Regional/Results-SeaweedRegionalSearch2024.csv')
    return
    #parse_docx('data/papers/2024-Seaweed-NexisUni/NexisUni_Seaweed_2024.DOCX')
    #parse_docx('data/papers/Aquaculture-NexisUni-2024/NexisUni_Aquaculture_2024_1-100.DOCX')
    #parse_docx('data/papers/Aquaculture-NexisUni-2024/NexisUni_Aquaculture_2024_1-100.DOCX')
    #parse_docx('data/papers/2024-Seaweed-NexisUni/NexisUni_Seaweed_2024.DOCX')
    #return

    doc_path_list = [
        pdf_file_path
        for pdf_file_glob in args.path_to_files
        for pdf_file_path in Path().rglob(pdf_file_glob)]

    for doc_path in doc_path_list:
        if doc_path.lower().endswith('.docx'):
            parse_docx(doc_path)
        elif doc_path.lower().endswith('.rtf'):
            parse_rtf(doc_path)
        elif doc_path.lower().endswith('.csv'):
            parse_csv(doc_path)

    return

    init_db()
    db = SessionLocal()

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
