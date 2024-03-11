"""Tracer code to figure out how to parse out DocX files."""
from transformers import pipeline

from database_model_definitions import Article, AIResultBody, USER_CLASSIFIED_BODY_OPTIONS
from database import SessionLocal, init_db

MODEL_PATH = 'wwf-seaweed-body-subject/allenai-longformer-base-4096_13'
BODY_LABEL_TO_SENTIMENT = {
    f'LABEL_{index}': label
    for index, label in enumerate(USER_CLASSIFIED_BODY_OPTIONS)
}


def main():
    print(f'load {MODEL_PATH}')
    headline_sentiment_model = pipeline(
        'sentiment-analysis', model=MODEL_PATH, device='cuda', truncation=True)
    print('loaded...')
    init_db()
    session = SessionLocal()
    bodies_without_ai = [
        article.body for article in
        session.query(Article).outerjoin(AIResultBody, Article.id_key == AIResultBody.article_id)
        .filter(
            AIResultBody.id_key == None,
            Article.body != None,
            Article.body != '')
        .all()]
    print(f'doing sentiment-analysis on {len(bodies_without_ai)} headlines')
    headline_sentiment_result_list = [
        {
            'label': BODY_LABEL_TO_SENTIMENT[val['label']],
            'score': val['score']
        } for val in headline_sentiment_model(bodies_without_ai)]
    print(headline_sentiment_result_list)
    print(bodies_without_ai)
    print('updating headline sentiment in database')
    for body, headline_sentiment_result in zip(
            bodies_without_ai, headline_sentiment_result_list):
        print(body)
        same_body_articles = session.query(Article).filter(
            Article.body == body).all()
        for article in same_body_articles:
            article.body_subject_ai = [
                AIResultBody(
                    value=headline_sentiment_result['label'],
                    score=headline_sentiment_result['score'])]

    session.commit()
    session.close()


if __name__ == '__main__':
    main()
