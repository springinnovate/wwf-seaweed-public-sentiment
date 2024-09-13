import glob
import re
import time

from database_model_definitions import Article, UrlOfArticle
from database_operations import upsert_articles
from database import SessionLocal, init_db
from database_model_definitions import AQUACULTURE_RE, SEAWEED_RE


TITLE_RE = r'"title": \["(.*?)"\]'
URL_RE = r'"url": \["(.*?)"\]'
EMBEDDED_DOMAIN_RE = r"(?:.*?http://){2}([^/]+)/"


def main():
    article_count = 0
    seaweed_count = 0
    aquaculture_count = 0
    both_count = 0
    init_db()
    db = SessionLocal()
    file_list = (
        list(glob.glob(
            'data/susseccsulresults_seaweed_regional_search_update_2015_2023/*.json')) +
        list(glob.glob(
            'data/successfulresults_seaweed_regional_search/*.json')))

    for json_file in file_list:
        article_list = []
        print(json_file)
        with open(json_file, encoding='utf-8', errors='ignore') as file:
            for line in file:
                line = line.strip()
                if line.startswith('"url"'):
                    print(line)
                    match = re.search(URL_RE, line)
                    url_text = match.group(1) if match else None
                    print(url_text)
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
                        body_text = ' '.join(
                            eval(line.split('"paragraph": ')[1]))
                        new_article = db.query(Article).filter(
                            Article.headline == headline_text,
                            Article.body == body_text,
                            Article.date == formatted_date,
                            Article.publication == body_text,
                            Article.source_file == json_file,
                        ).first()
                        if not new_article:
                            new_article = Article(
                                headline=headline_text,
                                body=body_text,
                                date=formatted_date,
                                publication=body_text,
                                source_file=json_file,
                                ground_truth_body_subject=None,
                                ground_truth_body_location=None,
                                )
                        if url_text:
                            match = re.search(EMBEDDED_DOMAIN_RE, url_text)
                            article_source_domain = (
                                match.group(1) if match else None)
                            new_article.url_of_article = UrlOfArticle(
                                raw_url=url_text,
                                article_source_domain=article_source_domain)
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

    #db.commit()
    #db.close()
    print('all done')


if __name__ == '__main__':
    main()
