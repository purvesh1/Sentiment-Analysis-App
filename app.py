from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import folium  
from streamlit_folium import folium_static
import numpy as np
from src.sentimentAnalysisApp.pipeline.prediction import PredictionPipeline
from src.sentimentAnalysisApp.constants import *

class ReviewPredictionMatrix:
    def __init__(self, aspects, sentiments):
        self.aspects = aspects
        self.sentiments = sentiments
        # Initialize a matrix of -1s
        self.matrix = np.full((len(aspects), len(sentiments)), 0)
        
    def update_entry(self, aspect, sentiment, prediction):
        # Update the matrix based on the provided aspect, sentiment, and prediction value.
        row_idx = self.aspects.index(aspect)
        col_idx = self.sentiments.index(sentiment)
        self.matrix[row_idx, col_idx] = prediction

    def update_from_list(self, list_of_predictions):
        """
        Updates the matrix from a list of predictions.
        :param list_of_predictions: List of format [(tuple: ( tuple : aspect, sentiment), binary_prediction)]
        """
        for (aspect, sentiment), prediction in list_of_predictions:
            self.update_entry(aspect, sentiment, prediction)
    
    def get_matrix(self):
        return self.matrix
    
    def visualize(self):
        # Visualize the matrix for this review, e.g., using a heatmap.
        pass

# maintaining both orders of aspects and sentiments
#aspects = ['price', 'anecdotes', 'food', 'ambience', 'service']
#sentiments = ['positive', 'neutral', 'negative', 'conflict', 'none']

# def aggregate_to_matrix(series_of_tuples):
    
#     # Create a matrix object
#     matrix_obj = ReviewPredictionMatrix(aspects, sentiments)
    
#     # Loop over the series and update the matrix
#     for absa_tuple in series_of_tuples:
#         matrix_obj.update_entry(absa_tuple[0][0], absa_tuple[0][1], absa_tuple[1])
    
#     return matrix_obj.get_matrix()

# final_df = pd.read_csv('C:/Users/91909/Documents/Term 3/Machine Learning/assignment/senti/Sentiment-Analysis-App/artifacts/mergedabsa.csv', encoding = 'latin-1')
# # def string_to_tuple(s):
# #     return tuple(s.strip("()").split("-"))

# final_df['combinations'] = final_df['combinations'].apply(eval)

# final_df['combo_label_pairs'] = list(zip(final_df['combinations'], final_df['predicted_labels']))

# # Step 2: Group by 'reviews' and aggregate the tuples
# aggregated_pairs = final_df.groupby('reviews')['combo_label_pairs'].apply(aggregate_to_matrix).reset_index()

# # Step 3: Merge with the original dataframe
# merged_with_aggregation = final_df.drop(columns=['combinations', 'predicted_labels', 'combo_label_pairs']).drop_duplicates()
# df = merged_with_aggregation.merge(aggregated_pairs, on='reviews')

# Display content based on the navigation choice:
def aggregate_matrices_for_location(location):
    
    # subset = df[df['store_address'] == location]

    # # Extract the actual matrices from the combo_label_pairs column
    # matrices = subset['combo_label_pairs']
    # #matrices = matrices.apply(lambda x: np.array(x) if not isinstance(x, np.ndarray) else x)
    # string_entries = matrices[matrices.apply(lambda x: isinstance(x, str))]

    # subset_stack = np.stack(matrices)
    # # Stack and sum them
    # aggregate_matrix = np.sum(subset_stack, axis=0)
    
    # return aggregate_matrix
    return None


def plot_location_heatmap(location):
    aggregate_matrix = aggregate_matrices_for_location(location)
    plt.figure(figsize=(10,6))
    sns.heatmap(aggregate_matrix, annot=True, cmap='YlGnBu', xticklabels=sentiments, yticklabels=aspects)
    plt.title(f"Sentiment Distribution for Location: {location}")
    st.pyplot()


# Define the data fetch function (placeholder for now)
def fetch_data():
    # Placeholder function to fetch data
    # Ideally, this would fetch data from your database or other sources
    return None

# Create a function for each page/section:

def show_home():
    # Your main visualization for the homepage here
    st.header("Analyse your sentiment 💬")
    st.write("Enter a sentence about your dining experience, and we'll analyze the sentiments for different aspects like food, service, and ambiance.")

    # query
    user_question = st.text_input("Try it out:")

    prediction_pipeline = PredictionPipeline()
    # add a button to run the model
    if st.button("Analyse"):
        # run the model
        answer = prediction_pipeline.predict(user_question)
        # print the answer and the score in red
        st.write(f"Sentiment: {answer}")
    # (Include the code to generate and display the heatmap here.)

def show_time_series():
    st.title('Sentiment Trends Over Time')
    # Your time series visualization here
    st.write('Interactive line chart showing sentiment over time.')
    # (Include the code to generate and display the time series chart here.)

def show_location_analysis():
    st.title('Location-based Sentiment Analysis')
    # Your location-based sentiment map visualization here
    st.write('Interactive map showing sentiments across different locations.')
    # (Include the code to generate and display the sentiment map here.)
    # add dropdown to select location
    #location = st.selectbox('Select a location', df.store_address.unique())
    # plot heatmap for selected location
    #plot_location_heatmap(location)

def show_detailed_insights():
    st.title('Deep Dive Into Aspects')
    # Dropdown to select specific aspects
    aspect = st.selectbox('Select an aspect for deeper insights', aspects)
    # Depending on the aspect selected, show relevant visuals and insights.
    st.write(f'Insights for {aspect}')
    
    # location
    #locations = df[['latitude ', 'longitude', 'store_address', 'rating']].dropna().drop_duplicates()
    # rename columns to match the format
    #locations.columns = ['lat', 'lon', 'store_address', 'rating']

    # print(locations.head())
    # # create a folium map
   
    # # Display the folium map using folium_static
    # m = folium.Map()

    # # Inside the loop where you add markers to the map
    # for index, row in locations.iterrows():
    #     lat = row['lat']
    #     lon = row['lon']
    #     address = row['store_address']
    #     # from row['rating'] get the numerical rating and strip the extra characters
    #     rating = np.round(float(row['rating'].split(' ')[0]),2)
    #     tooltip = f"Address: {address}\nRating: {rating}"
    #         # Generate the pie chart HTML code for the specific store
    #     # popup_html = generate_pie_chart(address, rating, positive, negative, neutral)
        
    #     # # Create a popup with the pie chart HTML code
    #     # popup = folium.Popup(popup_html, max_width=400)

    #     # Add a marker for the store with the popup
    #     folium.Marker(location=[lat, lon], tooltip=tooltip, popup=popup).add_to(m)

    # # Display the folium map using folium_static
    #     st.markdown('### Map of McDonald\'s Stores')
    #     folium_static(m)
        
    # (Include the code to generate and display visuals for the selected aspect.)

def show_about():
    st.title('About This Dashboard')
    st.write('This dashboard provides insights into customer sentiments based on reviews from Google Places API. It leverages BERT-based aspect sentiment analysis to derive detailed sentiments on various aspects of McDonald’s services and products.')

def main():
    load_dotenv()
    # Set the page title and layout
    st.set_page_config(page_title="McDonald's Sentiment Analysis", layout='wide', initial_sidebar_state='collapsed')

    # Sidebar for Navigation
    st.sidebar.title('Navigation')
    section = st.sidebar.radio('Go to', ['Home', 'Time Series Analysis', 'Location Analysis', 'Detailed Insights', 'About'])

        
    if section == 'Home':
        show_home()
    elif section == 'Time Series Analysis':
        show_time_series()
    elif section == 'Location Analysis':
        show_location_analysis()
    elif section == 'Detailed Insights':
        show_detailed_insights()
    elif section == 'About':
        show_about()

    
    
        
    
    # # wordcloud
    # # st.markdown('### Wordcloud')
    # # plot_wordcloud(df['review'], title="Word Cloud of Reviews")
    

    # # Show the reviews
    # st.markdown('### Reviews')
    # #st.write(df['review'])


if __name__ == '__main__':
    main()
