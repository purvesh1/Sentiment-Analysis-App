from sentimentAnalysisApp.config.configuration import ConfigurationManager
from sentimentAnalysisApp.components.data_transformation import DataTransformation
from transformers import BertTokenizer
from scipy.special import softmax
import numpy as np
from transformers import pipeline


class ABSA_PredictDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])  # Assuming 'input_ids' is a key in encodings

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self,text):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)

        if isinstance(input_data, str):
            input_data = [input_data]
        
        reviews_list, combinations_list = data_transformation.add_ABS_list(input_data)

        tokenizer = BertTokenizer.from_pretrained(self.config.tokenizer_save_spath)
        model = BertForSequenceClassification.from_pretrained(self.config.model_save_path)
        model.eval()

        trainer = Trainer(
                model=model
                )

        test_encodings = tokenizer(reviews_list, combinations_list, truncation=True, padding=True, return_tensors="pt")
        test_predict_dataset = ABSA_PredictDataset(test_encodings)
        
        results = trainer.predict(test_predict_dataset)

        scores = [softmax(prediction) for prediction in results.predictions]

        # Get the predicted labels
        predicted_labels = [np.argmax(x) for x in scores]

        # Convert predictions and scores to a DataFrame for visualization
        df_results = pd.DataFrame({
            'Predicted_Label': predicted_labels,
            'Scores': scores
        })

        # Display the first few rows of the DataFrame
        df_results.head()

        print("Dialogue:")
        print(text)

        output = df[df['Predicted_Label'] == 1]
        print("\nPredicted_Labels:")
        print(output)

        return output