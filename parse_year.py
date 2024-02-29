"""Parse out the year part of the date and add that."""
import re

from database_model_definitions import Article
from database import SessionLocal, init_db


def main():
    init_db()
    session = SessionLocal()
    for article in session.query(Article).all():
        if article.date is not None:
            year = re.search('(\d{4})', article.date)
            if year:
                article.year = int(year.group())
    session.commit()
    session.close()


if __name__ == '__main__':
    main()
