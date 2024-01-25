"""Tracer code to figure out how to parse out DocX files."""
import argparse
import glob
import time

from docx import Document


def read_docx(file_path):
    class Parser:
        def __init__(self):
            doc = Document(file_path)
            self.iter = iter(doc.paragraphs)

        def __iter__(self):
            return self

        def __next__(self):
            while True:
                val = next(self.iter).text.strip()
                if val != '':
                    return val

    paragaph_iter = Parser()
    for text in paragaph_iter:
        if text != '':
            headline = text
            break
    for text in paragaph_iter:
        if text == 'Body':
            break

    body_text = ''
    for text in paragaph_iter:
        if text == 'Classification' or text.startswith('----------------'):
            break
        body_text += text

    custom_tags = {
        'Subject': '',
        'Industry': '',
        'Geographic': '',
        'Load-Date': '',
    }

    for text in paragaph_iter:
        tag = text.split(':')[0]
        if tag in custom_tags:
            custom_tags[tag] = text
    return {
        'Body': body_text
    }.update(custom_tags)


def main():
    parser = argparse.ArgumentParser(description='parse docx')
    parser.add_argument('path_to_files', help='Path/wildcard to docx files')
    args = parser.parse_args()

    start_time = time.time()
    for index, file_path in enumerate(glob.glob(args.path_to_files)):
        print(f'parsing {index} {file_path}')
        print(f'avg time = {(time.time()-start_time)/(index+1)}')
        read_docx(file_path)


if __name__ == '__main__':
    main()
