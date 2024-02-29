"""Script to clean the DB as we figure out what that means."""
from database_model_definitions import Article
from database import SessionLocal, init_db


def main():
    init_db()
    session = SessionLocal()
    session.query(Article).filter(
        Article.body == '').delete()
    session.commit()

    for article in session.query(Article).all():
        cleaned_headline = article.headline.replace('\n', ' ').strip()
        article.headline = cleaned_headline
        cleaned_body = article.body.replace('\n', ' ').strip()
        article.body = cleaned_body
        print(cleaned_headline)

    session.commit()
    session.close()


if __name__ == '__main__':
    main()
