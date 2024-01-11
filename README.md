**Training the sentiment model**

To train, build a headline table modeled after `data/papers/froelich_headlines.csv` and run

`python headline_sentiment_analysis_trainer.py --headline_table_path PATH_TO_TABLE` 

Each epoch will be logged to the terminal and training will halt when 5 iterations of training have not been lower than the minimum loss seen so far in training. Model epoch checkpoints will be in `./wwf-seaweed-headline-sentiment/microsoft-deberta-v3-base_{epoch}` and a model training summary will be in the root directory at `modelperform.csv`.

**Conduct sentiment analysis on data**

Select a trained model checkpoint from the process above and run 

`python headline_sentiment_analysis_trainer.py --test_only --model_checkpoint_paths ./wwf-seaweed-headline-sentiment/microsoft-deberta-v3-base_{EPOCH} --headline_table_path PATH_TO_TABLE`

Results will be in a table in the root directory at `microsoft-deberta-v3-base_{EPOCH}_results.csv`.
