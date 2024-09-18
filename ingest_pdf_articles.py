"""Tracer code to figure out how to parse out DocX files."""
from collections import deque
from pathlib import Path
import argparse
import concurrent.futures
import io
import re
import sys
import time

from pypdf import PdfReader
from concurrent.futures import ProcessPoolExecutor

from database_model_definitions import Article
from database_operations import upsert_articles
from database import SessionLocal, init_db

# match the headline by finding the date
ACCESS_WORLDNEWS_RE = '(.*(?:\n[^\n]+)+)\n\n(\S* \d{1,2}, \d{4}) \| (.*)'


def parse_pdf(file_path):
    start_time = time.time()
    print(f'parsing {file_path}')

    with open(file_path, 'rb') as f:
        content = f.read()

    # Find the start of the PDF header (%PDF)
    pdf_start = content.find(b'%PDF')

    if pdf_start == -1:
        raise ValueError("Could not find valid PDF header")

    # Create an in-memory bytes stream from the valid PDF portion
    valid_pdf_content = content[pdf_start:]
    pdf_stream = io.BytesIO(valid_pdf_content)

    # Use PdfReader to read from the in-memory byte stream
    reader = PdfReader(pdf_stream)

    print('we got the reader')
    articles_text = ''
    for page_index, page in enumerate(reader.pages):
        articles_text += page.extract_text(extraction_mode="layout")+'\n\n'
    body_end_index = len(articles_text)
    new_article_list = []

    found_access_world_news = False
    for match in reversed([match for match in re.finditer(ACCESS_WORLDNEWS_RE, articles_text)]):
        found_access_world_news = True
        headline_text, date_text, publication_text = match.groups()
        body_text = articles_text[match.span()[1]:body_end_index]
        headline_text = ' '.join(headline_text.split('\n')).strip()
        body_end_index = match.span()[0]
        new_article = Article(
            headline=headline_text,
            body=body_text,
            date=date_text,
            publication=publication_text,
            source_file=str(file_path),
        )
        new_article_list.append(new_article)
    if not found_access_world_news:
        # must be NexisUni style:
        text_iter = iter(articles_text.splitlines())
        state = 'outside'
        for line in text_iter:
            line = line.strip()
            if line == '':
                continue
            #print(f'{state}: {line}')
            if state == 'outside':
                if line == 'End of Document':
                    state = 'eod'
            elif state == 'eod':
                if line == 'Bibliography':
                    state = 'outside'
                else:
                    state = 'title'
            if state == 'title':
                headline_text = next(text_iter).strip()
                #print(f'HEADLINE: {headline_text}')
                last_2_lines = deque(maxlen=2)
                while not any(line.startswith(month) for month in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]):
                    #print(f'not publication: {line}')
                    line = next(text_iter).strip()
                    last_2_lines.append(line)
                date_text = last_2_lines.pop()
                publication_text = last_2_lines.pop()
                #print(f'publication_text: "{publication_text}"')
                #print(f'date_text: {date_text}')
                while line != 'Body':
                    #print(f'not body: {line}')
                    line = next(text_iter).strip()
                body_text = ''
                line = next(text_iter).strip()
                while line != 'End of Document':
                    if line != '':
                        body_text += line
                    line = next(text_iter).strip()
                new_article = Article(
                    headline=headline_text,
                    body=body_text,
                    date=date_text,
                    publication=publication_text,
                    source_file=str(file_path),
                )
                new_article_list.append(new_article)
                state = 'eod'
    print(f'time to parse {file_path} = {time.time()-start_time:.1f}s')
    return new_article_list


def main():
    parser = argparse.ArgumentParser(description='parse pdf')
    parser.add_argument('path_to_files', nargs='+', help='Path/wildcard to pdf files')
    args = parser.parse_args()

    pdf_file_path_list = [
        pdf_file_path
        for pdf_file_glob in args.path_to_files
        for pdf_file_path in Path().rglob(pdf_file_glob)]

    init_db()
    db = SessionLocal()

    with ProcessPoolExecutor() as executor:
        future_list = {
            executor.submit(parse_pdf, file_path): file_path for file_path in pdf_file_path_list}

        # Iterate over futures to collect results and handle exceptions
        article_list = []
        for future in concurrent.futures.as_completed(future_list):
            file_path = future_list[future]
            try:
                # Get the result of the future
                articles = future.result()
                article_list.extend(articles)
            except Exception as e:
                # Log the error and continue
                with open('ingest_pdf_error_log.txt', 'a') as error_file:
                    error_file.write(f'on {file_path}; Error processing file: {e}\n')

    upsert_articles(db, article_list)
    db.commit()
    db.close()


if __name__ == '__main__':
    main()
