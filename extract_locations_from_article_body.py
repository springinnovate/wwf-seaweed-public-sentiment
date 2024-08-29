"""Extract geographic locations from article body."""
from transformers import pipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer

from database_model_definitions import Article, AIResultBody, USER_CLASSIFIED_BODY_OPTIONS, RELEVANT_SUBJECT_TO_LABEL, AQUACULTURE_SUBJECT_TO_LABEL, SEAWEED_LABEL, OTHER_AQUACULTURE_LABEL
from database import SessionLocal, init_db

import torch


def main():

    # Load the model and tokenizer
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Function to predict and extract locations
    def extract_locations(text):
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = inputs.tokens()

        # Predict
        with torch.no_grad():
            outputs = model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)
        print(predictions)
        print(tokens)

        # Extract locations
        locations = []
        for token, prediction in zip(tokens, predictions[0].numpy()):
            if model.config.id2label[prediction] in ['B-LOC', 'I-LOC']:  # B-LOC and I-LOC are typically used for locations
                locations.append(token)

        return locations


if __name__ == '__main__':
    main()
