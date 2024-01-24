"""Tracer code to figure out how to parse out DocX files."""
import argparse
import glob

from docx import Document

def read_docx(file_path):
    doc = Document(file_path)
    for para in doc.paragraphs:
        print(para.text)


def main():
    parser = argparse.ArgumentParser(description='parse docx')
    parser.add_argument('path_to_files', help='Path/wildcard to docx files')
    args = parser.parse_args()

    for file_path in glob.glob(args.path_to_files):
        print(f'parsing {file_path}')
        read_docx(file_path)
        return
