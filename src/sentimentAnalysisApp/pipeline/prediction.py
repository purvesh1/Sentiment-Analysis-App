from sentimentAnalysisApp.config.configuration import ConfigurationManager
from sentimentAnalysisApp.components.data_transformation import DataTransformation
from sentimentAnalysisApp.logging import logger
from sentimentAnalysisApp.entity import ModelEvaluationConfig

from scipy.special import softmax
import numpy as np
import pandas as pd

import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# class ABSA_PredictDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         return item

#     def __len__(self):
#         return len(self.encodings['input_ids'])  # Assuming 'input_ids' is a key in encodings

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self,input_data):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        model_evaluation_config = config.get_model_evaluation_config()

        if isinstance(input_data, str):
            input_data = [input_data]
        
        reviews_list, combinations_list = data_transformation.add_ABS_list(input_data)
        tokenizer = BertTokenizer.from_pretrained(model_evaluation_config.tokenizer_save_path)
        
        model = TFBertForSequenceClassification.from_pretrained(model_evaluation_config.model_save_path)

        test_encodings = tokenizer(reviews_list, combinations_list, truncation=True, padding=True, return_tensors="tf")
        test_predict_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings),))
        batch_size = 32
        test_predict_dataset = test_predict_dataset.batch(batch_size)
        predictions = model.predict(test_predict_dataset)
        probabilities = tf.nn.softmax(predictions[0], axis=-1).numpy()
        predicted_labels = np.argmax(probabilities, axis=-1)

        # Convert predictions and scores to a DataFrame for visualization
        df_results = pd.DataFrame({
            'Predicted_Label': predicted_labels
        })

        print("Dialogue:")
        print(input_data)

        output = [combinations_list[i] for i in df_results[df_results['Predicted_Label'] == 1].index.tolist()]

        print("\nPredicted_Labels:")
        print(output)

        return output