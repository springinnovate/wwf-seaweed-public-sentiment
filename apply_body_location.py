"""Tracer code to figure out how to parse out DocX files."""
from transformers import pipeline
import re

from database_model_definitions import Article, AIResultLocation
from database import SessionLocal, init_db

import textwrap


with open('global_locations.txt', 'r') as location_file:
    GLOBAL_LOCATIONS = [x.strip().lower() for x in location_file.readlines()]


def extract_locations_large_text(ner_pipeline, text, max_length=512, stride=256):
    chunks = textwrap.wrap(
        text, width=max_length, subsequent_indent=' ',
        break_long_words=False, break_on_hyphens=False)
    word_set = set()
    location_list = []
    for chunk in chunks:
        ner_results = ner_pipeline(chunk)

        locations = [
            entity['word'] for entity in ner_results
            if entity['entity'] in ['I-LOC']
            and not entity['word'].startswith('##')
            and not entity['word'] in word_set
            and len(entity['word']) > 1]
        word_set.update(locations)
        location_list.extend(locations)
    location_str = ' '.join(location_list)
    global_location_list = []
    for global_location in GLOBAL_LOCATIONS:
        if global_location in location_str.lower():
            print(f'{global_location} from {location_str}')
            global_location_list.append(global_location)
    return ';'.join(global_location_list)


def main():
    ner_pipeline = pipeline(
        "ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
        device='cuda')
    init_db()
    session = SessionLocal()
    articles_without_location = (
        session.query(Article).outerjoin(
            AIResultLocation, Article.id_key == AIResultLocation.article_id)
        .filter(
            AIResultLocation.id_key == None,
            Article.body != None,
            Article.body != '')
        .all())
    bodies_without_location = [
        article.body for article in articles_without_location]
    print(
        f'doing location analysis on {len(bodies_without_location)} '
        f'article bodies')
    location_result_list = [
        extract_locations_large_text(ner_pipeline, body)
        for body in bodies_without_location
        ]

    print('updating database')
    for article, location_result in zip(
            articles_without_location, location_result_list):
        article.geographic_location_ai = [
            AIResultLocation(
                value=location_result,
                score=1)]
    session.commit()
    session.close()


if __name__ == '__main__':
    main()
