"""Database definitions for news articles and their classifications."""
from sqlalchemy import Date
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from typing import List
from typing import Optional


class Base(DeclarativeBase):
    pass


class Article(Base):
    __tablename__ = 'article'
    id_key: Mapped[int] = mapped_column(primary_key=True)
    subject: Mapped[Optional[str]]
    industry: Mapped[Optional[str]]
    geographic: Mapped[Optional[str]]
    headline: Mapped[str]
    body: Mapped[str]
    publication: Mapped[str]
    date: Mapped[Date]
    headline_sentiment_ai: Mapped[Optional[str]]
    headline_sentiment_ai_score: Mapped[Optional[float]]
    body_subject_inferred_ai: Mapped[Optional[str]]
    location_inferred_ai: Mapped[Optional[List["AIResult"]]] = relationship(
        back_populates="article")


class AIResult(Base):
    __tablename__ = "ai_result"
    id_key: Mapped[int] = mapped_column(primary_key=True)
    article_id = mapped_column(ForeignKey("article.id_key"))
    value: Mapped[str]
    score: Mapped[float]
    article: Mapped[Article] = relationship(back_populates="ai_result")
