"""Tracer code to search a subreddit."""
import logging
import sys

import praw


USER_AGENT = 'python:wwf-seaweed-sentiment-bot:0.0 (by /u/wwf-seaweed-sentiment-bot)'


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def main():
    with open('secrets.txt', 'r') as secret_file:
        client_id, client_secret = secret_file.read().split('\n')

    LOGGER.debug(f'client: {client_id}, secret: {client_secret}')

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=f'python:{client_id}:0.0 (by /u/{client_id})'
    )

    # Define the subreddit and the search query
    subreddit_name = 'Maine'
    search_query = 'seaweed'

    # Access the subreddit
    subreddit = reddit.subreddit(subreddit_name)

    # Search the subreddit for the query and fetch titles and dates
    for submission in subreddit.search(search_query, limit=10):
        LOGGER.info(
            f'TITLE: {submission.title}, DATETIME: {submission.created_utc}')

if __name__ == '__main__':
    main()
