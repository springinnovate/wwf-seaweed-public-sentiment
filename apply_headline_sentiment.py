"""Tracer code to figure out how to parse out DocX files."""
from transformers import pipeline

from database_model_definitions import Article, AIResultHeadline
from database import SessionLocal, init_db
from sqlalchemy import distinct

MODEL_PATH = 'wwf-seaweed-headline-sentiment/microsoft-deberta-v3-base_6'
HEADLINE_LABEL_TO_SENTIMENT = {
    'LABEL_0': 'bad',
    'LABEL_1': 'neutral',
    'LABEL_2': 'good',
}


def main():
    print(f'load {MODEL_PATH}')
    headline_sentiment_model = pipeline(
        'sentiment-analysis', model=MODEL_PATH, device='cuda')
    print('loaded...')
    init_db()
    session = SessionLocal()
    headlines = [
        result[0] for result in
        session.query(distinct(Article.headline)).all()]
    print(headlines)
    print('doing sentiment-analysis')
    headline_sentiment_result = headline_sentiment_model(headlines)

    print('updating headline sentiment in database')
    for headline, headline_sentiment_result in zip(
            headlines, headline_sentiment_result):
        print(headline)
        headline_sentiment_entry = AIResultHeadline(
            value=HEADLINE_LABEL_TO_SENTIMENT[
                headline_sentiment_result['label']],
            score=headline_sentiment_result['score'])
        same_headline_articles = session.query(Article).filter(
            Article.headline == headline).all()
        for article in same_headline_articles:
            article.headline_sentiment_ai = headline_sentiment_entry

    session.commit()
    session.close()


if __name__ == '__main__':
    main()
