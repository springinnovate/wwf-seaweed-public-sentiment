import glob
import re
import time

from database_model_definitions import Article
from database_operations import upsert_articles
from database import SessionLocal, init_db


AQUACULTURE_RE = "(?i)aquaculture|(?i)g aquaculture|(?i)offshore aquaculture"
SEAWEED_RE = "((?i)seaweed|(?i)kelp|(?i)sea moss) .* ((?i)aquaculture|(?i)farm*|(?i)cultivat*)"


AQUACULTURE_RE = re.compile(
    'aquaculture|offshore aquaculture', re.IGNORECASE)
SEAWEED_RE = re.compile(
    '(seaweed|kelp|sea moss) .* (aquaculture|farm*|cultivat*)', re.IGNORECASE)

TITLE_RE = r'"title": \["(.*?)"\]'
URL_RE = r'"url": \["(.*?)"\]'

def main():
    article_count = 0
    seaweed_count = 0
    aquaculture_count = 0
    both_count = 0
    init_db()
    db = SessionLocal()
    for json_file in glob.glob(
            'data/successfulresults_seaweed_regional_search/*.json'):
        article_list = []
        with open(json_file, encoding='utf-8', errors='ignore') as file:
            for line in file:
                line = line.strip()
                if line.startswith('"url"'):
                    match = re.search(URL_RE, line)
                    url_text = match.group(1) if match else None
                    raw_date = url_text.split('/web/')[1][:8]
                    formatted_date = (
                        f"{raw_date[:4]}/{raw_date[4:6]}/{raw_date[6:]}")
                if line.startswith('"title"'):
                    match = re.search(TITLE_RE, line)
                    headline_text = match.group(1) if match else None
                if line.startswith('"paragraph"'):
                    article_count += 1
                    match_count = 0
                    if re.search(SEAWEED_RE, line):
                        seaweed_count += 1
                        match_count += 1
                    if re.search(AQUACULTURE_RE, line):
                        aquaculture_count += 1
                        match_count += 1
                    if match_count == 2:
                        both_count += 1
                    if match_count > 0:
                        body_text = ' '.join(eval(line.split('"paragraph": ')[1]))
                        new_article = Article(
                            headline=headline_text,
                            body=body_text,
                            date=formatted_date,
                            publication=body_text,
                            source_file=json_file,
                            ground_truth_body_subject=None,
                            ground_truth_body_location=None,
                            )
                        article_list.append(new_article)
                        body_text = None
                        headline_text = None
                        formatted_date = None
            print(f'inserting {json_file}')
            upsert_articles(db, article_list)
            print(
                f'articles: {article_count}\n'
                f'seaweed: {seaweed_count}\n'
                f'aquaculture: {aquaculture_count}\n'
                f'both: {both_count}\n')
    db.commit()
    db.close()
    print('all done')


if __name__ == '__main__':
    main()
