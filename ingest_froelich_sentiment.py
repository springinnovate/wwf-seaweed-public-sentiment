"""Inject Froelich summaries"""
import pandas

from database_model_definitions import Article
from database_operations import upsert_articles
from database import SessionLocal, init_db
from sqlalchemy import func

MODEL_PATH = 'wwf-seaweed-headline-sentiment/microsoft-deberta-v3-base_6'
HEADLINE_LABEL_TO_SENTIMENT = {
    -1: 'bad',
    0: 'neutral',
    1: 'good',
}


def main():
    init_db()
    session = SessionLocal()
    print('updating headline sentiment in database')

    punctuations = ['.', ',', '!', '?', ';', ':', '-', '_']

    # Starting point for our filter
    filter_expression = func.lower(Article.headline)

    # Dynamically build the filter expression to remove punctuation
    for p in punctuations:
        filter_expression = func.replace(filter_expression, p, '')

    froelich_table = pandas.read_csv("data/papers/froelich_headlines.csv")
    new_article_list = []
    for index, row in froelich_table.iterrows():
        # Do the same for the search string
        search_string = row['headline'].lower()
        froelich_sentiment = HEADLINE_LABEL_TO_SENTIMENT[row['sentiment']]
        for p in punctuations:
            search_string = search_string.replace(p, '')

        matching_articles = session.query(Article).filter(
            func.lower(Article.headline) == func.lower(row['headline'])).all()
        for article in matching_articles:
            article.ground_truth_headline_sentiment = froelich_sentiment
            article.year = row['year']
            article.ground_truth_body_subject = row['field']
            article.ground_truth_body_location = row['region']

        if matching_articles == []:
            # insert a new article
            new_article = Article(
                headline=row['headline'],
                year=row['year'],
                ground_truth_headline_sentiment=froelich_sentiment,
                source_file='data/papers/froelich_headlines.csv',
                ground_truth_body_subject=row['field'],
                ground_truth_body_location=row['region'],
                )
            new_article_list.append(new_article)

    upsert_articles(session, new_article_list)
    session.commit()
    session.close()


if __name__ == '__main__':
    main()
