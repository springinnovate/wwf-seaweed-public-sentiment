"""Tracer code to figure out how to parse out DocX files."""
import argparse
import glob
import re
import time

from pypdf import PdfReader
from concurrent.futures import ThreadPoolExecutor

from database_model_definitions import Article
from database import SessionLocal, init_db

RE_TEXT = '(.*(?:\n[^\n]+)+)\n\n(\S* \d{1,2}, \d{4}) \| (.*)'


def parse_pdf(file_path):
    start_time = time.time()
    print(f'parsing {file_path}')
    reader = PdfReader(file_path)
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

        print(f'time to parse = {time.time()-start_time}s')
        print(result)

        new_article = Article(
            headline=headline_text,
            body=body_text,
            date=date_text,
            publication=publication_text
            )
        new_article_list.append(new_article)
    return new_article_list


def main():
    parser = argparse.ArgumentParser(description='parse pdf')
    parser.add_argument('path_to_files', help='Path/wildcard to pdf files')
    args = parser.parse_args()

    init_db()
    db = SessionLocal()

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_list = []
        for index, file_path in enumerate(glob.glob(args.path_to_files)):
            future = executor.submit(parse_pdf, file_path)
            future_list.append(future)
            break
        article_list = [article for future in future_list for article in future.result()]
    db.add_all(article_list)
    db.commit()
    db.close()


if __name__ == '__main__':
    main()
