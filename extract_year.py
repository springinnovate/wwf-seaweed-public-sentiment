"""Tracer code to figure out how to parse out DocX files."""
import io
import concurrent.futures
from pathlib import Path
import argparse
import re
import time

from pypdf import PdfReader
from concurrent.futures import ProcessPoolExecutor

from database_model_definitions import Article
from database_operations import upsert_articles
from database import SessionLocal, init_db
from sqlalchemy.orm import Session
from sqlalchemy import update
from datetime import datetime
import re

year_regex = re.compile(r'(\d{4})')


def extract_year_from_date(date_str: str) -> int:
    match = year_regex.search(date_str)
    if match:
        return int(match.group(0))
    return None


def update_article_years(session: Session):
    articles = session.query(Article).all()
    update_data = []
    for article in articles:
        if article.year is not None:
            continue
        if article.date:
            year = extract_year_from_date(article.date)
            if year:
                update_data.append({
                    'id_key': article.id_key,
                    'year': year
                })

    # Bulk update the year field
    if update_data:
        session.bulk_update_mappings(Article, update_data)
    else:
        print('nothing to dupate')


def main():
    init_db()
    db = SessionLocal()
    update_article_years(db)
    db.commit()
    db.close()


if __name__ == '__main__':
    main()
