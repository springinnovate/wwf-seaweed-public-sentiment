"""Tracer code to figure out how to parse out DocX files."""
import io
import concurrent.futures
from pathlib import Path
import argparse
import re
import time

from pypdf import PdfReader
from concurrent.futures import ProcessPoolExecutor

from database_model_definitions import Article
from database_operations import upsert_articles
from database import SessionLocal, init_db

RE_TEXT = '(.*(?:\n[^\n]+)+)\n\n(\S* \d{1,2}, \d{4}) \| (.*)'


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
    for match in reversed([match for match in re.finditer(RE_TEXT, articles_text)]):
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
