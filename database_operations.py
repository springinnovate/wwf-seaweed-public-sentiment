"""Database definitions for news articles and their classifications."""
from database_model_definitions import Article
from typing import List

def upsert_articles(session, article_list: List[Article]):
    for new_article in article_list:
        # Check if an article with the same headline, body, date, and publication exists
        existing_article = session.query(Article).filter(
            Article.headline == new_article.headline,
            Article.body == new_article.body,
            Article.date == new_article.date,
            Article.publication == new_article.publication
        ).first()

        if existing_article:
            # Update existing article
            existing_article.year = new_article.year
            existing_article.headline_sentiment_ai = new_article.headline_sentiment_ai
            existing_article.body_subject_ai = new_article.body_subject_ai
            existing_article.geographic_location_ai = new_article.geographic_location_ai
            existing_article.ground_truth_headline_sentiment = new_article.ground_truth_headline_sentiment
            existing_article.ground_truth_body_subject = new_article.ground_truth_body_subject
            existing_article.ground_truth_body_location = new_article.ground_truth_body_location
            existing_article.source_file = new_article.source_file
        else:
            # Add as a new article
            session.add(new_article)

    session.commit()