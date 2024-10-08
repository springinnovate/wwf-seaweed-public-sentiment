# WWF Seaweed Media Sentiment Pipeline

## Table of Contents
- [Environment](#environment)
  - [Execution Environment](#execution-environment)
  - [Database Structure](#database-structure)
    - [Article Table](#article-table)
    - [AIResultHeadline Table](#airesultheadline-table)
    - [AIResultBody Table](#airesultbody-table)
    - [AIResultLocation Table](#airesultlocation-table)
    - [UrlOfArticle Table](#urlofarticle-table)
- [User Facing Scripts](#user-facing-scripts)
  - [Headline Sentiment Analysis](#headline-sentiment-analysis)
    - [Headline Sentiment Active Learning Pipeline](#headline-sentiment-active-learning-pipeline)
    - [Headline Sentiment Classification Pipeline](#headline-sentiment-classification-pipeline)
  - [Article Subject Classification](#article-subject-classification)
    - [Article Subject Active Learning Pipeline](#article-subject-active-learning-pipeline)
    - [Article Subject Classification Pipeline](#article-subject-classification-pipeline)
    - [Article Body Location Classification](#article-body-location-classification)
  - [Report Generation](#report-generation)

## Enviroment

### Execution Environment

The code in this repository requires several machine learning Python dependencies, including `pytorch`, the HuggingFace library components, `scikit-learn`, `ninja`, `flash-attention`, and others. Some of these dependencies involve complex compilation and configuration steps that can take hours of computation. To simplify the execution process, a precompiled Docker image is available at `therealspring/convei_abstract_classifier:latest`.

You can run any of the Python scripts in this repository within the interactive Docker environment using the following command:

`docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v .:/workspace therealspring/convei_abstract_classifier:latest`

### Database Structure

The database structure is designed to store and manage information related to articles, headlines, their associated metadata, and the results of AI-based sentiment and topic classifications. The primary components are defined in `database_model_definitions.py` `database_operations.py` as follows:

#### Article Table
- **ID Key**: Unique identifier for each article.
- **Headline**: The title or main heading of the article.
- **Body**: The full text of the article (optional).
- **Date**: The publication date of the article (optional).
- **Year**: The year of publication (optional).
- **Publication**: The name of the publication where the article appeared (optional).
- **URL of Article**: Relationship to a table storing the article's URL and its source domain.
- **Headline Sentiment AI**: Relationship to AI-generated sentiment analysis results for the headline.
- **Body Subject AI**: Relationship to AI-generated subject classifications for the body of the article.
- **Geographic Location AI**: Relationship to AI-generated geographic location classifications for the article.
- **Ground Truth Headline Sentiment**: Manually classified sentiment of the headline (optional).
- **Ground Truth Body Subject**: Manually classified subject of the body (optional).
- **Ground Truth Body Location**: Manually classified location information of the body (optional).
- **User Classified Body Subject**: User-provided classification of the article's subject (optional).
- **User Classified Body Location**: User-provided classification of the article's location information (optional).
- **Source File**: The source file from which the article was extracted (optional).

#### AIResultHeadline Table
- **ID Key**: Unique identifier for each AI-generated headline sentiment result.
- **Article ID**: Foreign key linking to the associated article.
- **Value**: The sentiment value classified by the AI.
- **Score**: Confidence score of the AI classification.

#### AIResultBody Table
- **ID Key**: Unique identifier for each AI-generated body subject classification result.
- **Article ID**: Foreign key linking to the associated article.
- **Value**: The subject classification value provided by the AI.
- **Score**: Confidence score of the AI classification.

#### AIResultLocation Table
- **ID Key**: Unique identifier for each AI-generated geographic location classification result.
- **Article ID**: Foreign key linking to the associated article.
- **Value**: The geographic location classified by the AI.
- **Score**: Confidence score of the AI classification.

#### UrlOfArticle Table
- **ID Key**: Unique identifier for each URL record.
- **Article ID**: Foreign key linking to the associated article.
- **Raw URL**: The raw URL of the article.
- **Article Source Domain**: The domain name from which the article was sourced.

## User Facing Scripts

### Headline Sentiment Analysis

The first stage of this project involved classifying the sentiment of seaweed-related headlines from the dataset provided in "`Froelich et al. 2017.pdf`". This paper served as the source for both training and validation data. After the initial model was developed, an active learning cycle was implemented to refine the model with additional data.

This section includes three key components:

- **Parser**: `ingest_froelich_sentiment.py`
  - Responsible for parsing the sentiment data from "Froelich et al. 2017.pdf" and preparing it for use in training and validation.

- **Trainer**: `trainer_headline_sentiment.py`
  - Used to train the headline sentiment model based on the parsed data. This script facilitates the training process using the data from the Froelich paper as well as additional data gathered through active learning.

- **Application**: `apply_headline_sentiment.py`
  - This script applies the trained sentiment model to new headlines. The model that is used in practice is located at `SOMEWHERE ONLINE THAT SAM WILL FIGURE OUT`. The model should be downloaded and extracted into this repository under the path `wwf-seaweed-headline-sentiment/microsoft-deberta-v3-base_7`. This path is referenced in the `apply_headline_sentiment.py` pipeline for sentiment analysis.

These components work together to process, train, and apply sentiment analysis on seaweed-related headlines, ensuring that the model is continuously improved and accurately reflects the sentiment expressed in the headlines.

#### Headline Sentiment Active Learning Pipeline

The active learning pipeline for training the headline sentiment model is managed through the script `trainer_headline_sentiment.py`. This script is executed using the command:

`python trainer_headline_sentiment.py --headline_table_path PATH_TO_HEADLINE_TABLE`

The `PATH_TO_HEADLINE_TABLE` should point to a table containing the headers `headline` and `label`, where `headline` represents the text of the article headlines and `label` represents the sentiment classification 0 (negative), 1 (neutral), or 2 (positive).

As the script runs, it will produce a series of models, each stored in a subdirectory within `wwf-seaweed-headline-sentiment`. The performance of each model iteration is logged in a table called `modelperform.csv`, located in the root directory of the repository.

Users can inspect the `modelperform.csv` table to review the performance metrics of the different model iterations and choose the iteration that best meets their criteria for application. This allows for flexible selection and further refinement of the model based on the active learning results.

#### Headline Sentiment Classification Pipeline

Once articles have been ingested into the database using the `ingest_froelich_sentiment.py` script, the sentiment model can be applied to unclassified headlines with this command:

`python apply_headline_sentiment.py`

This script automatically applies the trained sentiment model from the previous section to all unclassified headlines in the database. By default, it works directly with the database, but the script can be easily modified to accept input from other sources, such as a CSV table, allowing for flexibility in how the sentiment analysis is applied to different datasets.

### Article Subject Classification

News article data were provided in both Microsoft Word documents and PDF files, with formats varying based on their source. We provide parsers for these formats in the scripts `ingest_docx_articles.py`, `parser_regional_articles.py`, and `ingest_pdf_articles.py`. These scripts populate the same SQLite database used for headline sentiment analysis, and they also support the article body topic, sentiment, and location mapping described in the sections below.

#### Article Subject Active Learning Pipeline

Article subject training was conducted in two phases: the first phase classified articles as relevant or irrelevant, and the second phase further classified relevant articles into subjects related to seaweed aquaculture and other types of aquaculture.

The pipeline begins with the script `python user_validates_subject.py`, which allows the user to review and classify unclassified articles in the database created during the ingestion step. The user selects the appropriate classification, and the process continues iteratively.

Once the database contains a sufficient number of classified articles, users can run `python trainer_two_phase_body_subject.py` to train both the relevant/irrelevant classifier and the subject-specific classifier. The resulting models are stored in directories named `wwf-seaweed-body-subject-{aquaculture-type|relevant-irrelevant}`.

This process is typically repeated based on the performance of the model during the classification stage, allowing for iterative improvements.

#### Article Subject Classification Pipeline

Once the models are built, the body subjects can be classified using the script `python apply_body_subject_classification.py`. By default, this classification process utilizes the pretrained models `wwf-seaweed-body-subject-relevant-irrelevant/allenai-longformer-base-4096_19` and `wwf-seaweed-body-subject-aquaculture-type/allenai-longformer-base-4096_36`, which can be downloaded from `SOMEWHERE ONLINE THAT SAM WILL FIGURE OUT`.

No additional user input is required; the pipeline will automatically classify all article bodies in the database created during the ingestion step.

#### Article Body Location Classification

To classify the global locations mentioned in the article bodies, you can use the script `apply_body_location.py`. This script utilizes the `dbmdz/bert-large-cased-finetuned-conll03-english` model to perform location analysis on the text of the articles. The results are automatically updated in the database.

### Report Generation

We provide the `build_reports.py` script as an example of how article results can be displayed, analyzed, and post-processed for various purposes. This script generates goodness-of-fit confusion matrices, tracks sentiment and subject counts over time, and calculates other statistics useful for analysis.

Users are encouraged to modify this script to suit their specific needs or to use it as a reference for post-processing the results generated by the pipelines.
