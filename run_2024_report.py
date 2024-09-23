"""
 *   # relevant aquaculture, and
     # relevant seaweed aquaculture articles
        * global then broken out by maine, new Hampshire, Massachusetts, Alaska, Washington, Oregon, california
 *   % positive, negative, neutral sentiment for other aquaculture –
        * global, then broken out by maine, new Hampshire, Massachusetts, Alaska, Washington, Oregon, california
 *   % positive, negative, neutral sentiment for seaweed aquaculture –
        *  global, then broken out by maine, new Hampshire, Massachusetts, Alaska, Washington, Oregon, California
"""
from datetime import datetime
from collections import defaultdict

from database import SessionLocal
from database_model_definitions import Article, AIResultBody, AIResultHeadline, AIResultLocation
from sqlalchemy import select
from sqlalchemy.orm import joinedload
import pandas as pd

LOCAL_LOCATIONS = ['maine', 'new hampshire', 'massachusetts', 'alaska', 'washington', 'oregon', 'california']


def main():
    session = SessionLocal()
    query = (
        select(Article)
        .join(AIResultBody)
        .join(AIResultHeadline, Article.headline_sentiment_ai)
        .join(AIResultLocation, Article.geographic_location_ai)
        .options(
            joinedload(Article.body_subject_ai),
            joinedload(Article.headline_sentiment_ai),
            joinedload(Article.geographic_location_ai)
        )
        .where(
            Article.year == 2024,
            AIResultBody.value != 'IRRELEVANT'
        )
    )

    result = session.scalars(query).unique().all()

    relevant_location_count = defaultdict(int)
    relevant_location_count_by_geography = defaultdict(lambda: defaultdict(int))
    sentiment_count_by_aquaculture_type = defaultdict(lambda: defaultdict(int))
    sentiment_count_by_aquaculture_type_by_geography = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for article in result:
        for body in article.body_subject_ai:
            relevant_location_count[body.value] += 1
            sentiment_count_by_aquaculture_type[body.value][article.headline_sentiment_ai.value] += 1
            for local_location in LOCAL_LOCATIONS:
                if local_location in article.geographic_location_ai[0].value:
                    relevant_location_count_by_geography[local_location][body.value] += 1
                    sentiment_count_by_aquaculture_type_by_geography[local_location][body.value][article.headline_sentiment_ai.value] += 1
            print(f"Article ID: {article.id_key}, Article Classification: {body.value}, Headline Value: {article.headline_sentiment_ai.value}, Location Value: {article.geographic_location_ai[0].value if article.geographic_location_ai else 'N/A'}")

    df_relevant_location_count = pd.DataFrame(
        list(relevant_location_count.items()),
        columns=['Article Classification', 'Article Count']
    )

    print("Count of Articles by Classification:")
    print(df_relevant_location_count.to_string(index=False))
    table1 = "Count of Articles by Classification:\n"
    table1 += df_relevant_location_count.to_csv(index=False)

    df_relevant_location_count_by_geography = pd.DataFrame(relevant_location_count_by_geography).fillna(0).astype(int)
    df_relevant_location_count_by_geography = df_relevant_location_count_by_geography.transpose()
    df_relevant_location_count_by_geography.reset_index(inplace=True)
    df_relevant_location_count_by_geography.rename(columns={'index': 'Article Classification'}, inplace=True)
    print("\nCount of Articles by Article Classification and Geography:")
    print(df_relevant_location_count_by_geography.to_string(index=False))
    table2 = "\n\nCount of Articles by Article Classification and Geography:\n"
    table2 += df_relevant_location_count_by_geography.to_csv(index=False)

    df_sentiment_count_by_aquaculture_type = pd.DataFrame(sentiment_count_by_aquaculture_type).fillna(0).astype(int)
    df_sentiment_count_by_aquaculture_type = df_sentiment_count_by_aquaculture_type.transpose()
    df_sentiment_count_by_aquaculture_type.reset_index(inplace=True)
    df_sentiment_count_by_aquaculture_type.rename(columns={'index': 'Article Classification'}, inplace=True)
    print("\nCount of Headline Sentiment by Aquaculture Type:")
    print(df_sentiment_count_by_aquaculture_type.to_string(index=False))
    table3 = "\n\nCount of Headline Sentiment by Aquaculture Type:\n"
    table3 += df_sentiment_count_by_aquaculture_type.to_csv(index=False)

    data = []
    for geography, body_dict in sentiment_count_by_aquaculture_type_by_geography.items():
        for body_value, sentiment_dict in body_dict.items():
            for sentiment, count in sentiment_dict.items():
                data.append({
                    'Geography': geography,
                    'Article Classification': body_value,
                    'Headline Sentiment': sentiment,
                    'Article Count': count
                })
    df_sentiment_geography = pd.DataFrame(data)
    pivot_table = df_sentiment_geography.pivot_table(
        index=['Geography', 'Article Classification'],
        columns='Headline Sentiment',
        values='Article Count',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    pivot_table.columns.name = None
    pivot_table.columns = [col if not isinstance(col, tuple) else col[1] for col in pivot_table.columns]
    print("\nCount of Headline Sentiment by Aquaculture Type and Geography:")
    print(pivot_table.to_string(index=False))
    table4 = "\n\nCount of Headline Sentiment by Aquaculture Type and Geography:\n"
    table4 += pivot_table.to_csv(index=False)

    output = table1 + table2 + table3 + table4
    print(output)

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    filename = f'aquaculture_media_report_{timestamp}.csv'
    with open(filename, 'w', newline='') as f:
        f.write(output)


if __name__ == '__main__':
    main()
