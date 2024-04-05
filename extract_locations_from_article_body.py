"""Tracer code to figure out how to parse out DocX files."""
from transformers import pipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer

from database_model_definitions import Article, AIResultBody, USER_CLASSIFIED_BODY_OPTIONS, RELEVANT_SUBJECT_TO_LABEL, AQUACULTURE_SUBJECT_TO_LABEL, SEAWEED_LABEL, OTHER_AQUACULTURE_LABEL
from database import SessionLocal, init_db

import torch



def main():

    # Load the model and tokenizer
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Function to predict and extract locations
    def extract_locations(text):
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = inputs.tokens()

        # Predict
        with torch.no_grad():
            outputs = model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)

        # Extract locations
        locations = []
        for token, prediction in zip(tokens, predictions[0].numpy()):
            if model.config.id2label[prediction] in ['B-LOC', 'I-LOC']:  # B-LOC and I-LOC are typically used for locations
                locations.append(token)

        return " ".join(locations)

    # Example usage
    article_text = "The quick brown fox jumps over the lazy dog in New York."
    extracted_locations = extract_locations(article_text)
    print(extracted_locations)








    # relevant_subject_model = pipeline(
    #     'text-classification', model=RELEVANT_SUBJECT_MODEL_PATH,
    #     device='cuda', truncation=True)
    # aquaculture_subject_model = pipeline(
    #     'text-classification', model=AQUACULTURE_SUBJECT_MODEL_PATH,
    #     device='cuda', truncation=True)
    # print('loaded models...')
    # init_db()
    # session = SessionLocal()
    # bodies_without_ai = [
    #     article.body for article in
    #     session.query(Article).outerjoin(AIResultBody, Article.id_key == AIResultBody.article_id)
    #     .filter(
    #         AIResultBody.id_key == None,
    #         Article.body != None,
    #         Article.body != '')
    #     .all()]
    # print(f'doing sentiment-analysis on {len(bodies_without_ai)} headlines')
    # relevant_subject_result_list = [
    #     {
    #         'label': RELEVANT_LABEL_TO_SUBJECT[val['label']],
    #         'score': val['score']
    #     } for val in relevant_subject_model(bodies_without_ai)]

    # relevant_bodies = [
    #     body for body, classification in
    #     zip(bodies_without_ai, relevant_subject_result_list)
    #     if classification['label'] == 'RELEVANT']

    # aquaculture_type_result_list = [
    #     {
    #         'label': AQUACULTURE_LABEL_TO_SUBJECT[val['label']],
    #         'score': val['score']
    #     } for val in aquaculture_subject_model(relevant_bodies)]

    # print('updating database')
    # for body, headline_sentiment_result in zip(
    #         bodies_without_ai, relevant_subject_result_list):
    #     if headline_sentiment_result['label'] == 'RELEVANT':
    #         continue
    #     print(body)
    #     same_body_articles = session.query(Article).filter(
    #         Article.body == body).all()
    #     for article in same_body_articles:
    #         article.body_subject_ai = [
    #             AIResultBody(
    #                 value=headline_sentiment_result['label'],
    #                 score=headline_sentiment_result['score'])]
    # for body, headline_sentiment_result in zip(
    #         aquaculture_type_result_list, relevant_bodies):
    #     print(body)
    #     same_body_articles = session.query(Article).filter(
    #         Article.body == body).all()
    #     for article in same_body_articles:
    #         article.body_subject_ai = [
    #             AIResultBody(
    #                 value=headline_sentiment_result['label'],
    #                 score=headline_sentiment_result['score'])]

    # session.commit()
    # session.close()


if __name__ == '__main__':
    main()
