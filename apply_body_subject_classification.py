"""Tracer code to figure out how to parse out DocX files."""
import os
from transformers import pipeline
import re

from database_model_definitions import Article, AIResultBody, SEAWEED_LABEL, OTHER_AQUACULTURE_LABEL
from database_model_definitions import RELEVANT_LABEL, IRRELEVANT_LABEL
from database_model_definitions import RELEVANT_TAG, IRRELEVANT_TAG, SEAWEED_TAG, OTHER_AQUACULTURE_TAG
from database import SessionLocal, init_db


RELEVANT_LABEL_TO_SUBJECT = {
    f'LABEL_{RELEVANT_LABEL}': RELEVANT_TAG,
    f'LABEL_{IRRELEVANT_LABEL}': IRRELEVANT_TAG
}

AQUACULTURE_LABEL_TO_SUBJECT = {
    f'LABEL_{SEAWEED_LABEL}': SEAWEED_TAG,
    f'LABEL_{OTHER_AQUACULTURE_LABEL}': OTHER_AQUACULTURE_TAG,
}

RELEVANT_SUBJECT_MODEL_PATH = "wwf-seaweed-body-subject-relevant-irrelevant/allenai-longformer-base-4096_19"
AQUACULTURE_SUBJECT_MODEL_PATH = "wwf-seaweed-body-subject-aquaculture-type/allenai-longformer-base-4096_36"


def standardize_text(text):
    text = re.sub(r'[!?]', '.', text)
    text = re.sub(r'[\'",;:-]', '', text)
    text = text.replace('\n', ' ').replace('\r', '')
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def make_pipeline(model_type, model_path):
    _pipeline = pipeline(
        model_type, model=model_path,
        device='cuda', truncation=True)

    def _standardize_and_sentiment(raw_text_list):
        clean_text = [
            standardize_text(raw_text)
            for raw_text in raw_text_list]
        unique_text = list(set(clean_text))
        text_to_index_lookup = {
            text_val: index for index, text_val in enumerate(unique_text)
        }
        unique_results = _pipeline(unique_text)
        full_results = [
            unique_results[text_to_index_lookup[base_text_val]]
            for base_text_val in clean_text]
        return full_results

    return _standardize_and_sentiment


def main():
    not_found = False
    for model_path in [RELEVANT_SUBJECT_MODEL_PATH, AQUACULTURE_SUBJECT_MODEL_PATH]:
        if not os.path.exists(model_path):
            not_found = True
            print(f'{model_path} not found, you need to download it from wherever Sam uploaded it to, ask her!')
    if not_found:
        return

    seaweed_re = re.compile('(seaweed|kelp|sea moss)', re.IGNORECASE)
    relevant_subject_model = make_pipeline(
        'text-classification', RELEVANT_SUBJECT_MODEL_PATH)
    aquaculture_subject_model = make_pipeline(
        'text-classification', AQUACULTURE_SUBJECT_MODEL_PATH)
    print('loaded models...')
    init_db()
    session = SessionLocal()
    articles_without_ai = (
        session.query(Article).outerjoin(
            AIResultBody, Article.id_key == AIResultBody.article_id)
        .filter(
            AIResultBody.id_key == None,
            Article.body != None,
            Article.body != '')
        .all())
    bodies_without_ai = [article.body for article in articles_without_ai]
    print(f'doing sentiment-analysis on {len(bodies_without_ai)} article bodies')
    return
    relevant_subject_result_list = [
        {
            'label': RELEVANT_LABEL_TO_SUBJECT[val['label']],
            'score': val['score']
        } for val in relevant_subject_model(bodies_without_ai)]

    relevant_articles = [
        article for article, classification in
        zip(articles_without_ai, relevant_subject_result_list)
        if classification['label'] == RELEVANT_TAG]
    relevant_bodies = [article.body for article in relevant_articles]
    aquaculture_type_result_list = [
        {
            'label': AQUACULTURE_LABEL_TO_SUBJECT[val['label']],
            'score': val['score']
        } for val in aquaculture_subject_model(relevant_bodies)]

    print('updating database')
    for article, relevant_subject_result in zip(
            articles_without_ai, relevant_subject_result_list):
        if relevant_subject_result['label'] == RELEVANT_TAG:
            continue
        article.body_subject_ai = [
            AIResultBody(
                value=relevant_subject_result['label'],
                score=relevant_subject_result['score'])]
    for article, aquaculture_type_result in zip(
            relevant_articles, aquaculture_type_result_list):
        working_label = aquaculture_type_result['label']
        if working_label == OTHER_AQUACULTURE_TAG and re.search(
                seaweed_re, article.body):
            # override if there's a regular expression match
            working_label = SEAWEED_TAG
        article.body_subject_ai = [
            AIResultBody(
                value=working_label,
                score=aquaculture_type_result['score'])]

    session.commit()
    session.close()


if __name__ == '__main__':
    main()
