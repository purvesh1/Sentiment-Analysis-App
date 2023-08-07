import os
import pandas as pd
from sentimentAnalysisApp.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from sentimentAnalysisApp.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def add_ABS_df(self,n):
        
        df = pd.read_csv(self.config.data_path)[:n]
        # Aspects and sentiments
        aspects = ['price', 'anecdotes', 'food', 'ambience', 'service']
        sentiments = ['positive', 'neutral', 'negative', 'conflict', 'none']

        combinations = [f"{aspect} - {sentiment}" for aspect in aspects for sentiment in sentiments]

        # Step 1: Add a column with all combinations for each row
        df['combinations'] = [combinations] * len(df)

        # Step 2: Explode the combinations into separate rows
        df = df.explode('combinations')

        # Reset the index for cleanliness
        df.reset_index(drop=True, inplace=True)
        df.to_csv(self.config.data_out_path,index=False)

    def add_ABS_list(self, reviews):
    # Aspects and sentiments
        aspects = ['price', 'anecdotes', 'food', 'ambience', 'service']
        sentiments = ['positive', 'neutral', 'negative', 'conflict', 'none']

        combinations_list = [f"{aspect} - {sentiment}" for aspect in aspects for sentiment in sentiments]

        # Create lists to hold the expanded reviews and combinations
        expanded_reviews = []
        expanded_combinations = []

        # For each review in the input, add all combinations
        for review in reviews:
            for combo in combinations_list:
                expanded_reviews.append(review)
                expanded_combinations.append(combo)

        return expanded_reviews, expanded_combinations


    