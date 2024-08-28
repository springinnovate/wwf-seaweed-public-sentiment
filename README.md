# WWF Seaweed Media Sentiment Pipeline

The database structure is designed to store and manage information related to articles, headlines, their associated metadata, and the results of AI-based sentiment and topic classifications. The primary components are defined in `database_model_definitions.py` `database_operations.py` as follows:

### Article Table
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

### AIResultHeadline Table
- **ID Key**: Unique identifier for each AI-generated headline sentiment result.
- **Article ID**: Foreign key linking to the associated article.
- **Value**: The sentiment value classified by the AI.
- **Score**: Confidence score of the AI classification.

### AIResultBody Table
- **ID Key**: Unique identifier for each AI-generated body subject classification result.
- **Article ID**: Foreign key linking to the associated article.
- **Value**: The subject classification value provided by the AI.
- **Score**: Confidence score of the AI classification.

### AIResultLocation Table
- **ID Key**: Unique identifier for each AI-generated geographic location classification result.
- **Article ID**: Foreign key linking to the associated article.
- **Value**: The geographic location classified by the AI.
- **Score**: Confidence score of the AI classification.

### UrlOfArticle Table
- **ID Key**: Unique identifier for each URL record.
- **Article ID**: Foreign key linking to the associated article.
- **Raw URL**: The raw URL of the article.
- **Article Source Domain**: The domain name from which the article was sourced.


## Headline Sentiment Analysis

### Active Learning Pipeline
Description of the active learning process for headline sentiment analysis.

### Headline Sentiment Classification Pipeline
Description of the classification pipeline used for determining sentiment in headlines.

## 2. Article Topic Classification

### 2a. Working Database Structure
Details on the database structure used for storing and managing article data.

### 2b. Parsing Documents
Explanation of the document parsing process for extracting relevant information from articles.

### 2c. Active Learning Pipeline
Description of the active learning process for article topic classification.

### 2d. Article Classification Pipeline
Details on the pipeline used for classifying articles into topics.





**Training the sentiment model**

To train, build a headline table modeled after `data/papers/froelich_headlines.csv` and run

`python headline_sentiment_analysis_trainer.py --headline_table_path PATH_TO_TABLE` 

Each epoch will be logged to the terminal and training will halt when 5 iterations of training have not been lower than the minimum loss seen so far in training. Model epoch checkpoints will be in `./wwf-seaweed-headline-sentiment/microsoft-deberta-v3-base_{epoch}` and a model training summary will be in the root directory at `modelperform.csv`.

**Conduct sentiment analysis on data**

Select a trained model checkpoint from the process above and run 

`python headline_sentiment_analysis_trainer.py --test_only --model_checkpoint_paths ./wwf-seaweed-headline-sentiment/microsoft-deberta-v3-base_{EPOCH} --headline_table_path PATH_TO_TABLE`

Results will be in a table in the root directory at `microsoft-deberta-v3-base_{EPOCH}_results.csv`.
