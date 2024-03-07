import glob
import time
import re

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from pygments.lexers import PythonLexer

from database_model_definitions import Article
from database_model_definitions import USER_CLASSIFIED_BODY_OPTIONS
from database_operations import upsert_articles
from database import SessionLocal, init_db
from sqlalchemy import func

OPTIONS = {
    index: key for index, key in enumerate(USER_CLASSIFIED_BODY_OPTIONS)
}


def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # Return None if value is not found in the dictionary


def print_options(options, selected):
    print("Options:")
    for i, option in options.items():
        print(f"{i}: {option} {'(selected)' if i in selected else ''}")


def get_user_choice(options, existing_subject):
    session = PromptSession()
    bindings = KeyBindings()
    selected = set()
    if existing_subject is not None:
        selected = set([
            get_key_from_value(OPTIONS, subject.strip())
            for subject in existing_subject.split(';')])
    choice = None

    @bindings.add('q')
    def _(event):
        nonlocal choice
        choice = 'quit'
        event.app.exit(result='c')

    @bindings.add('b')
    def _(event):
        nonlocal choice
        choice = 'back'
        event.app.exit(result=choice)

    def key_handler(event):
        nonlocal choice
        choice = int(event.data)
        if 0 <= choice < len(options):
            if choice in selected:
                selected.remove(choice)
            else:
                selected.add(choice)
            print("\033[F\033[K"*(len(OPTIONS)+1), end='')
            print_options(options, selected)

    for i in range(len(OPTIONS)):  # Add bindings for keys 1-9
        bindings.add(f"{i}")(key_handler)

    while True:
        try:
            print_options(options, selected)
            print("Press the number keys to select options, 'b' to go back, or press Enter to confirm: ", end='', flush=True)
            session.prompt("", key_bindings=bindings, default='')
            if isinstance(choice, str):
                return choice
            return [OPTIONS[index] for index in selected]
        except KeyboardInterrupt:
            return None


def highlight_keywords(body_text, keywords):
    highlighted_text = ""
    for word in body_text.split():
        for keyword in keywords:
            if keyword.lower() in word.lower():
                # Apply bold and highlighted background color
                highlighted_text += f"\033[1;48;5;231m{word}\033[0m "
                break
        else:
            highlighted_text += f"{word} "
    return highlighted_text


def main():

    init_db()
    session = SessionLocal()
    keywords = ['seaweed', 'aquaculture', 'kelp', 'fish', 'farm']

    last_article = None
    choice = None
    while True:
        if choice != 'back':
            existing_article = session.query(Article).filter(
                Article.user_classified_body_subject == None
            ).order_by(func.random()).first()
        body = existing_article.body
        if body is None:
            continue
        print(highlight_keywords(
            body.strip(), keywords))
        choice = get_user_choice(
            OPTIONS, existing_article.user_classified_body_subject)
        print(choice)
        if choice == 'quit':
            break
        if choice == 'back' and last_article:
            existing_article = last_article
            continue
        else:
            existing_article.user_classified_body_subject = ';'.join(choice)
        last_article = existing_article
        session.commit()

    session.close()


if __name__ == '__main__':
    main()
