"""Script to clean the DB as we figure out what that means."""
from database_model_definitions import Article
from database import SessionLocal, init_db


def main():
    init_db()
    session = SessionLocal()
    session.query(Article).filter(
        Article.body == '').delete()
    session.commit()
    session.close()


if __name__ == '__main__':
    main()
