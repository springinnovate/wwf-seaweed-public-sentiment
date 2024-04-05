"""Tracer code to figure out how to parse out DocX files."""
from transformers import pipeline

from database_model_definitions import Article, AIResultBody, USER_CLASSIFIED_BODY_OPTIONS, RELEVANT_SUBJECT_TO_LABEL, AQUACULTURE_SUBJECT_TO_LABEL, SEAWEED_LABEL, OTHER_AQUACULTURE_LABEL
from database_model_definitions import RELEVANT_LABEL, IRRELEVANT_LABEL
from database import SessionLocal, init_db


RELEVANT_LABEL_TO_SUBJECT = {
    f'LABEL_{RELEVANT_LABEL}': 'RELEVANT',
    f'LABEL_{IRRELEVANT_LABEL}': 'IRRELEVANT'
}

AQUACULTURE_LABEL_TO_SUBJECT = {
    f'LABEL_{SEAWEED_LABEL}': 'SEAWEED AQUACULTURE',
    f'LABEL_{OTHER_AQUACULTURE_LABEL}': 'OTHER AQUACULTURE'
}

RELEVANT_SUBJECT_MODEL_PATH = "wwf-seaweed-body-subject-relevant-irrelevant/allenai-longformer-base-4096_19"
AQUACULTURE_SUBJECT_MODEL_PATH = "wwf-seaweed-body-subject-aquaculture-type/allenai-longformer-base-4096_17"


def main():
    relevant_subject_model = pipeline(
        'text-classification', model=RELEVANT_SUBJECT_MODEL_PATH,
        device='cuda', truncation=True)
    aquaculture_subject_model = pipeline(
        'text-classification', model=AQUACULTURE_SUBJECT_MODEL_PATH,
        device='cuda', truncation=True)
    print('loaded models...')
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
    relevant_subject_result_list = [
        {
            'label': RELEVANT_LABEL_TO_SUBJECT[val['label']],
            'score': val['score']
        } for val in relevant_subject_model(bodies_without_ai)]

    relevant_bodies = [
        body for body, classification in
        zip(bodies_without_ai, relevant_subject_result_list)
        if classification['label'] == 'RELEVANT']

    aquaculture_type_result_list = [
        {
            'label': AQUACULTURE_LABEL_TO_SUBJECT[val['label']],
            'score': val['score']
        } for val in aquaculture_subject_model(relevant_bodies)]

    print('updating database')
    for body, relevant_subject_result in zip(
            bodies_without_ai, relevant_subject_result_list):
        if relevant_subject_result['label'] == 'RELEVANT':
            continue
        print(body)
        same_body_articles = session.query(Article).filter(
            Article.body == body).all()
        for article in same_body_articles:
            article.body_subject_ai = [
                AIResultBody(
                    value=relevant_subject_result['label'],
                    score=relevant_subject_result['score'])]
    for body, aquaculture_type_result in zip(
            relevant_bodies, aquaculture_type_result_list):
        print(body)
        same_body_articles = session.query(Article).filter(
            Article.body == body).all()
        for article in same_body_articles:
            article.body_subject_ai = [
                AIResultBody(
                    value=aquaculture_type_result['label'],
                    score=aquaculture_type_result['score'])]

    session.commit()
    session.close()


if __name__ == '__main__':
    main()
