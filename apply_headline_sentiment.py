"""Tracer code to figure out how to parse out DocX files."""
from transformers import pipeline

from database_model_definitions import Article, AIResultHeadline
from database import SessionLocal, init_db

MODEL_PATH = 'wwf-seaweed-headline-sentiment/microsoft-deberta-v3-base_7'
HEADLINE_LABEL_TO_SENTIMENT = {
    'LABEL_0': 'bad',
    'LABEL_1': 'neutral',
    'LABEL_2': 'good',
}


def main():
    if not os.path.exists(MODEL_PATH):
        print(f'{MODEL_PATH} not found, you need to download it from wherever Sam uploaded it to, ask her!')
        return
    print(f'load {MODEL_PATH}')
    headline_sentiment_model = pipeline(
        'sentiment-analysis', model=MODEL_PATH, device='cuda')
    print('loaded...')
    init_db()
    session = SessionLocal()
    articles_without_ai = (
        session.query(Article).outerjoin(AIResultHeadline, Article.id_key == AIResultHeadline.article_id)
        .filter(AIResultHeadline.id_key == None)
        .all())
    headlines_without_ai = [article.headline for article in articles_without_ai]

    print(f'doing sentiment-analysis on {len(headlines_without_ai)} headlines')
    headline_sentiment_result = headline_sentiment_model(headlines_without_ai)

    print('updating headline sentiment in database')
    for article, headline_sentiment_result in zip(
            articles_without_ai, headline_sentiment_result):
        article.headline_sentiment_ai = AIResultHeadline(
            value=HEADLINE_LABEL_TO_SENTIMENT[
                headline_sentiment_result['label']],
            score=headline_sentiment_result['score'])

    session.commit()
    session.close()


if __name__ == '__main__':
    main()
