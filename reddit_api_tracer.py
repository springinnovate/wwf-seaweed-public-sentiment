"""Tracer code to search a subreddit."""
import argparse
import logging
import sys

import praw

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Search Reddit')
    parser.add_argument('search_file')
    args = parser.parse_args()

    with open(args.search_file, 'r') as search_file:
        subreddit_list, search_query = search_file.read().rstrip().split('\n')
        subreddit_list = subreddit_list.split(' ')

    with open('secrets.txt', 'r') as secret_file:
        client_id, client_secret, username, password = \
            secret_file.read().rstrip().split('\n')

    LOGGER.debug(f'client: {client_id}, secret: {client_secret}')

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        password=password,
        username=username,
        user_agent=f'python:wwf-seaweed-sentiment-bot:0.0 (by /u/{username})'
    )

    # Define the subreddit and the search query
    for subreddit_name in subreddit_list:
        subreddit = reddit.subreddit(subreddit_name)
        for index, submission in enumerate(subreddit.search(
                search_query, sort='relevance', limit=100)):
            LOGGER.info(
                f'{index} -- TITLE: {submission.title}, DATETIME: {submission.created_utc}')


if __name__ == '__main__':
    main()
