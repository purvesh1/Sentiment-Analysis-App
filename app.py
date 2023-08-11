from dotenv import load_dotenv
import streamlit as st
import pandas as pd
# from box.exceptions import BoxValueError
# from sentimentAnalysisApp.utils.common import get_address

from sentimentAnalysisApp.pipeline.prediction import PredictionPipeline


st.set_option('deprecation.showPyplotGlobalUse', False)       


# Import necessary libraries
import streamlit as st


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

def show_detailed_insights():
    st.title('Deep Dive Into Aspects')
    # Dropdown to select specific aspects
    aspect = st.selectbox('Select an aspect for deeper insights', ['Fries', 'Service', 'Cleanliness', 'Ambience'])
    # Depending on the aspect selected, show relevant visuals and insights.
    st.write(f'Insights for {aspect}')
    # (Include the code to generate and display visuals for the selected aspect.)

def show_about():
    st.title('About This Dashboard')
    st.write('This dashboard provides insights into customer sentiments based on reviews from Google Places API. It leverages BERT-based aspect sentiment analysis to derive detailed sentiments on various aspects of McDonald’s services and products.')

# Display content based on the navigation choice:





def main():
    load_dotenv()
    # Set the page title and layout
    st.set_page_config(page_title="McDonald's Sentiment Analysis", layout='wide', initial_sidebar_state='collapsed')

    # Sidebar for Navigation
    st.sidebar.title('Navigation')
    section = st.sidebar.radio('Go to', ['Home', 'Time Series Analysis', 'Location Analysis', 'Detailed Insights', 'About'])

 
    df = pd.read_csv('artifacts/absa_matrix.csv', encoding='latin-1')

        
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
    # # load mcd dataset
    # df = pd.read_csv('data/merged_df.csv', encoding='latin-1')
    # df = df.loc[df['review'].str.contains(r'[^\x00-\x7F]+') == False]

    
    # # location
    # locations = df[['latitude ', 'longitude', 'store_address', 'rating']].dropna().drop_duplicates()
    # # rename columns to match the format
    # locations.columns = ['lat', 'lon', 'store_address', 'rating']

    # print(locations.head())
    # # create a folium map
   
    # # Display the folium map using folium_static
    # m = folium.Map()

    # # Inside the loop where you add markers to the map
    # for index, row in locations.iterrows():
    #     lat = row['lat']
    #     lon = row['lon']
    #     address = row['store_address']
    #     rating = np.round(row['rating'],2)
    #     positive = df.loc[df['store_address'] == address, 'positive'].iloc[0]
    #     negative = df.loc[df['store_address'] == address, 'negative'].iloc[0]
    #     neutral = df.loc[df['store_address'] == address, 'neutral'].iloc[0]
    #     tooltip = f"Address: {address}\nRating: {rating}"
        
    #     # Generate the pie chart HTML code for the specific store
    #     popup_html = generate_pie_chart(address, rating, positive, negative, neutral)
        
    #     # Create a popup with the pie chart HTML code
    #     popup = folium.Popup(popup_html, max_width=400)

    #     # Add a marker for the store with the popup
    #     folium.Marker(location=[lat, lon], tooltip=tooltip, popup=popup).add_to(m)

    # # Display the folium map using folium_static
    # st.markdown('### Map of McDonald\'s Stores')
    # folium_static(m)
    # # wordcloud
    # # st.markdown('### Wordcloud')
    # # plot_wordcloud(df['review'], title="Word Cloud of Reviews")
    

    # # Show the reviews
    # st.markdown('### Reviews')
    # #st.write(df['review'])


if __name__ == '__main__':
    main()
