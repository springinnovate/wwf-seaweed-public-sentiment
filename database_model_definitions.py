"""Database definitions for news articles and their classifications."""
from datetime import date
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
    headline: Mapped[str]
    body: Mapped[str]
    date: Mapped[date]
    publication: Mapped[str]
    headline_sentiment_ai: Mapped[List["AIResultHeadline"]] = relationship(back_populates="article")
    body_subject_ai: Mapped[Optional[List["AIResultBody"]]] = relationship(back_populates="article")
    geographic_location_ai: Mapped[Optional[List["AIResultLocation"]]] = relationship(back_populates="article")


class AIResultHeadline(Base):
    __tablename__ = "ai_result_headline"
    id_key: Mapped[int] = mapped_column(primary_key=True)
    article_id = mapped_column(ForeignKey("article.id_key"))
    value: Mapped[str]
    score: Mapped[float]
    article: Mapped[Article] = relationship(back_populates="headline_sentiment_ai")


class AIResultBody(Base):
    __tablename__ = "ai_result_body"
    id_key: Mapped[int] = mapped_column(primary_key=True)
    article_id = mapped_column(ForeignKey("article.id_key"))
    value: Mapped[str]
    score: Mapped[float]
    article: Mapped[Article] = relationship(back_populates="body_subject_ai")


class AIResultLocation(Base):
    __tablename__ = "ai_result_location"
    id_key: Mapped[int] = mapped_column(primary_key=True)
    article_id = mapped_column(ForeignKey("article.id_key"))
    value: Mapped[str]
    score: Mapped[float]
    article: Mapped[Article] = relationship(back_populates="geographic_location_ai")