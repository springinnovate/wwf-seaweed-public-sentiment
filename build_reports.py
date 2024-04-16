"""Do reports."""
import sys
import re
import collections
import string

import pandas
from database_model_definitions import Article, AIResultHeadline, AIResultBody, AIResultLocation, USER_CLASSIFIED_BODY_OPTIONS
from database_operations import filter_user_defined_body_to_top_level_tags
from database import SessionLocal, init_db
import matplotlib.pyplot as plt
from sqlalchemy import distinct
from sqlalchemy import select

def remove_non_printable(s):
    return ''.join(char for char in s if char.isprintable() and char != ',').encode('utf-8')


def main():
    init_db()
    session = SessionLocal()

    # Assuming you have your SQLAlchemy session named session and the models defined as Article, AIResultHeadline, AIResultBody, AIResultLocation
    query = (
        select(
            Article.headline,
            Article.date,
            Article.year,
            Article.source_file,
            AIResultHeadline.value.label('headline_value'),
            AIResultBody.value.label('body_value'),
            AIResultLocation.value.label('location_value')
        )
        .join(AIResultHeadline, Article.id_key == AIResultHeadline.article_id)
        .join(AIResultBody, Article.id_key == AIResultBody.article_id)
        .join(AIResultLocation, Article.id_key == AIResultLocation.article_id)
    )
    headline_set = set()
    sys.stdout.buffer.write(
            'headline,year,headline_sentiment,body_subject,location,search_type\n'.encode('utf-8'))

    for headline, date, year, source_file, headline_sentiment, body_subject, location in session.execute(query).all():
        # Use regular expression to find four consecutive digits
        if location in [' ', '']:
            location = None
        clean_headline = headline.lower().translate(
            str.maketrans('', '', string.punctuation))

        if clean_headline in headline_set:
            continue
        headline_set.add(clean_headline)
        if year is None and date is not None:
            match = re.search(r'\b\d{4}\b', date)

            # Extract the year if a match is found
            year = match.group(0) if match else None

        if 'seaweed_regional_search' in source_file:
            search_type = 'regional/internet archive'
        elif 'NexisUni' in source_file:
            search_type = 'NexisUni'
        elif 'NewsBank' in source_file:
            search_type = 'NewsBank'
        elif 'Froehlich' in source_file:
            search_type = 'Froehlich raw data'
        else:
            search_type = source_file
        sys.stdout.buffer.write(remove_non_printable(headline))

        sys.stdout.buffer.write(
            f', {year}, {headline_sentiment}, {body_subject.lower()}, {location}, {search_type}\n'.encode('utf-8'))
    return

    # build a AI subject body confusion matrix
    results = session.query(Article.user_classified_body_subject, AIResultBody.value).join(Article).filter(
        Article.user_classified_body_subject.isnot(None)).all()
    subject_confusion_matrix = collections.defaultdict(
        lambda: collections.defaultdict(int))

    for ground_truth_subject_str, classified_subject in results:
        primary_ground_truth = filter_user_defined_body_to_top_level_tags(
            ground_truth_subject_str)
        print(f'{ground_truth_subject_str} to {primary_ground_truth}')
        subject_confusion_matrix[primary_ground_truth][classified_subject] += 1
    print(subject_confusion_matrix)

    subject_confusing_matrix_path = 'subject_confusion_matrix.csv'
    with open(subject_confusing_matrix_path, 'w') as confusion_matrix_table:
        confusion_matrix_table.write(
            'modeled vs ground truth,' + ','.join(TOP_LEVEL_BODY_CLASSIFICATIONS) + '\n')
        for classified_subject in TOP_LEVEL_BODY_CLASSIFICATIONS:
            confusion_matrix_table.write(f'{classified_subject},')
            for index, ground_truth in enumerate(TOP_LEVEL_BODY_CLASSIFICATIONS):
                total = sum([
                    subject_confusion_matrix[ground_truth][local_modeled]
                    for local_modeled in TOP_LEVEL_BODY_CLASSIFICATIONS])
                count = subject_confusion_matrix[ground_truth][classified_subject]
                confusion_matrix_table.write(
                    f'{count} ({count/total*100:.1f}%)')
                if index < len(TOP_LEVEL_BODY_CLASSIFICATIONS)-1:
                    confusion_matrix_table.write(',')
            confusion_matrix_table.write('\n')

    plt.figure(figsize=(7, 5))

    # Load the confusion matrix CSV file into a DataFrame
    cm_subject_data = pandas.read_csv(subject_confusing_matrix_path, sep=',')
    print(cm_subject_data.head())
    # Extracting counts from the DataFrame
    # We'll use a regular expression to extract counts before the parentheses and convert them to integers
    cm_counts = cm_subject_data.apply(lambda x: x.str.extract('(\d+)')[0]).fillna(0).astype(int)

    # Dropping the first column since it's the labels for the rows
    cm_counts_values = cm_counts.iloc[:, 1:].values

    # Labels for the axes
    labels = cm_subject_data.columns[1:].tolist()  # Ground truth labels
    predicted_labels = cm_subject_data.iloc[:, 0].tolist()  # Modeled/predicted labels

    cm_percentages = cm_subject_data.apply(lambda x: x.str.extract('\((.*?)%\)', expand=False))
    cm_annotations = (cm_counts.iloc[:, 1:].astype(str) + " (" + cm_percentages.iloc[:, 1:] + "%)").values

    sns.heatmap(cm_counts_values, annot=cm_annotations, fmt="", cmap="Blues", xticklabels=labels, yticklabels=predicted_labels, cbar=False)
    plt.title('Confusion Matrix (Manually Classified vs Modeled)')
    plt.xlabel('Manually Classified')
    plt.ylabel('Modeled')
    plt.xticks(rotation=45, ha="center")
    plt.yticks(rotation=0)

    # Moving the ground truth labels to the top
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.show()

    return

    # Then report headline classifications over time based on
    # OTHER AQUACULTURE and SEAWEED AQUACULTURE
    for article in session.query(Article).filter(
            Article.user_classified_body_subject.isnot(None)).all():
        if article.headline in processed_headline_set:
            continue
        processed_headline_set.add(article.headline)
        confusion_matrix[
            article.ground_truth_headline_sentiment][
            article.headline_sentiment_ai.value] += 1
        data_dict['headline'].append(article.headline)
        data_dict['Froelich et al. 2017 sentiment'].append(
            article.ground_truth_headline_sentiment)
        data_dict['modeled sentiment'].append(
            article.headline_sentiment_ai.value)
        data_dict['modeled confidence'].append(
            article.headline_sentiment_ai.score)


    # Report headlines per subject type other aquacuture/seaweed aquaculture


    confusion_matrix = collections.defaultdict(
        lambda: collections.defaultdict(int))
    processed_headline_set = set()
    data_dict = {
        'headline': [],
        'Froelich et al. 2017 sentiment': [],
        'modeled sentiment': [],
        'modeled confidence': [],
        }

    for article in session.query(Article).filter(
            Article.ground_truth_headline_sentiment.isnot(None)).all():
        if article.headline in processed_headline_set:
            continue
        processed_headline_set.add(article.headline)
        confusion_matrix[
            article.ground_truth_headline_sentiment][
            article.headline_sentiment_ai.value] += 1
        data_dict['headline'].append(article.headline)
        data_dict['Froelich et al. 2017 sentiment'].append(
            article.ground_truth_headline_sentiment)
        data_dict['modeled sentiment'].append(
            article.headline_sentiment_ai.value)
        data_dict['modeled confidence'].append(
            article.headline_sentiment_ai.score)

    table = pandas.DataFrame(data_dict)
    table.to_csv('headline_sentiment_agreement.csv', index=False, columns=[
        'headline',
        'Froelich et al. 2017 sentiment',
        'modeled sentiment',
        'modeled confidence',
        ])
    with open('confusion_matrix.csv', 'w') as confusion_matrix_table:
        confusion_matrix_table.write(r'modeled vs ground truth,good,neutral,bad' + '\n')
        for modeled in ['good', 'neutral', 'bad']:
            confusion_matrix_table.write(f'{modeled},')
            for index, ground_truth in enumerate(['good', 'neutral', 'bad']):
                total = sum([
                    confusion_matrix[ground_truth][local_modeled]
                    for local_modeled in ['good', 'neutral', 'bad']])
                count = confusion_matrix[ground_truth][modeled]
                confusion_matrix_table.write(
                    f'{count} ({count/total*100:.1f}%)')
                if index < 2:
                    confusion_matrix_table.write(',')
            confusion_matrix_table.write('\n')

        # Creating the heatmap again with ground truth labels on top
    plt.figure(figsize=(7, 5))

    # Load the confusion matrix CSV file into a DataFrame
    cm_file_path = 'confusion_matrix.csv'
    cm_data = pandas.read_csv(cm_file_path)
    print(cm_data.head())
    # Extracting counts from the DataFrame
    # We'll use a regular expression to extract counts before the parentheses and convert them to integers
    cm_counts = cm_data.apply(lambda x: x.str.extract('(\d+)')[0]).fillna(0).astype(int)

    # Dropping the first column since it's the labels for the rows
    cm_counts_values = cm_counts.iloc[:, 1:].values

    # Labels for the axes
    labels = cm_data.columns[1:].tolist()  # Ground truth labels
    predicted_labels = cm_data.iloc[:, 0].tolist()  # Modeled/predicted labels

    cm_percentages = cm_data.apply(lambda x: x.str.extract('\((.*?)%\)', expand=False))
    cm_annotations = (cm_counts.iloc[:, 1:].astype(str) + " (" + cm_percentages.iloc[:, 1:] + "%)").values

    sns.heatmap(cm_counts_values, annot=cm_annotations, fmt="", cmap="Blues", xticklabels=labels, yticklabels=predicted_labels, cbar=False)
    plt.title('Confusion Matrix (Froelich 2017 vs Modeled)')
    plt.xlabel('Froelich 2017')
    plt.ylabel('Modeled')
    plt.xticks(rotation=45, ha="center")
    plt.yticks(rotation=0)

    # Moving the ground truth labels to the top
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.show()

    # Querying for distinct years in the Article table
    distinct_years = [
        result[0] for result in session.query(distinct(Article.year)).all()]
    print(distinct_years)

    data_dict = {
        'good': [],
        'neutral': [],
        'bad': [],
    }

    summary_dict = {
        'headline': [],
        'year': [],
        'body': [],
        'headline sentiment': [],
        'headline sentiment score': [],
        'source_file': [],
        'publication': [],
    }

    result_by_year_rows_list = []
    for year in sorted(distinct_years):
        print(year)
        result = session.query(Article).filter(Article.year == year).all()
        data_dict = collections.defaultdict(int)
        data_dict['year'] = year
        processed_headlines = set()
        for article in result:
            if article.headline in processed_headlines:
                continue
            processed_headlines.add(article.headline)
            data_dict[article.headline_sentiment_ai.value] += 1
            summary_dict['headline'].append(article.headline)
            summary_dict['year'].append(article.year)
            summary_dict['body'].append(article.body)
            summary_dict['headline sentiment'].append(article.headline_sentiment_ai.value)
            summary_dict['headline sentiment score'].append(article.headline_sentiment_ai.score)
            summary_dict['source_file'].append(article.source_file)
            summary_dict['publication'].append(article.publication)

        result_by_year_rows_list.append(data_dict)

    data_summary = pandas.DataFrame(summary_dict)
    data_summary.to_csv('headline_summary.csv', index=False, columns=[
        'headline',
        'year',
        'body',
        'headline sentiment',
        'headline sentiment score',
        'source_file',
        'publication',
        ])

    result_by_year = pandas.DataFrame(result_by_year_rows_list)
    result_by_year.to_csv(
        'sentiment_by_year.csv', index=False, columns=[
            'year',
            'good',
            'neutral',
            'bad',
            ])

    data_filled = result_by_year.fillna(0)

    # Calculate total sentiments per year for percentage calculation
    data_filled['total'] = data_filled[['good', 'neutral', 'bad']].sum(axis=1)

    # Calculate percentages
    data_filled['good_pct'] = data_filled['good'] / data_filled['total'] * 100
    data_filled['neutral_pct'] = data_filled['neutral'] / data_filled['total'] * 100
    data_filled['bad_pct'] = data_filled['bad'] / data_filled['total'] * 100

    # Preparing the plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # Plotting raw counts
    data_filled.plot(kind='bar', stacked=True, x='year', y=['good', 'neutral', 'bad'], ax=axes[0],
                     color={'good': 'green', 'neutral': 'gray', 'bad': 'red'})
    axes[0].set_title('Sentiment Counts by Year')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Count')
    axes[0].legend(title='Sentiment')
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)  # Adding gridlines to the percentages chart

    # Plotting percentages
    data_filled.plot(kind='bar', stacked=True, x='year', y=['good_pct', 'neutral_pct', 'bad_pct'], ax=axes[1],
                     color={'good_pct': 'green', 'neutral_pct': 'gray', 'bad_pct': 'red'})
    axes[1].set_title('Sentiment Percentages by Year')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Percentage')
    axes[1].legend(['Good', 'Neutral', 'Bad'], title='Sentiment')
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)  # Adding gridlines to the percentages chart

    plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Adding gridlines
    plt.tight_layout()
    plt.show()

    session.close()


if __name__ == '__main__':
    main()
