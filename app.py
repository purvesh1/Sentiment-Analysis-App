from dotenv import load_dotenv
import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from wordcloud import ImageColorGenerator, WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

st.set_option('deprecation.showPyplotGlobalUse', False)       

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), color = 'white',
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'u', "im"}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color=color,
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = None)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()

def predict_sentiment(text, model):
    model_dict = {
    'RoBERTa': 'cardiffnlp/twitter-roberta-base-sentiment',
    'DistilBERT': 'distilbert-base-uncased-finetuned-sst-2-english'
}

    if model not in model_dict:
        raise Exception(f"Model {model} not found!")
    
    # load the model
    tokenizer = AutoTokenizer.from_pretrained(model_dict[model])
    model = AutoModelForSequenceClassification.from_pretrained(model_dict[model])
    # Run for Roberta Model
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'Negative' : scores[0],
        'Neutral' : scores[1],
        'Positive' : scores[2]
    }
    # visualize the scores dict as a 
    # return key and the value for the max value
    return max(scores_dict, key=scores_dict.get), max(scores_dict.values())

def generate_pie_chart(address, rating, positive, negative, neutral):
    # Data for the pie chart
    sizes = [positive, negative, neutral]
    labels = ['Positive', 'Negative', 'Neutral']
    colors = ['green', 'red', 'blue']

    # Create the pie chart
    plt.figure(figsize=(3, 3))  # Adjust the size of the pie chart
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart appears as a circle

    # Add store name and rating at the bottom of the pie chart
    plt.text(-0.1, 1.3, f"{address}", ha='center', fontsize=7, weight='bold')  # Changed y-coordinate to 1.1
    plt.text(0.6, 1.15, f"Rating: {rating}", ha='center', fontsize=10, weight='bold')  # Changed y-coordinate to 1

    # Save the pie chart to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert the bytes buffer to a base64-encoded string
    pie_chart_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Generate the HTML code for the pie chart
    html = f'<img src="data:image/png;base64,{pie_chart_data}" />'
    
    return html


def get_address(lat, lon):
    geolocator = Nominatim(user_agent="sentiment_analysis_app")
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        return location.address
    except GeocoderTimedOut:
        return get_address(lat, lon)


def main():
    load_dotenv()

    st.set_page_config(page_title="Sentiment Analysis")
    st.header("Analyse your sentiment 💬")

    # add drop down menu
    st.sidebar.header("Menu")
    app_mode = st.sidebar.selectbox("Choose the model", ["RoBERTa", "DistilBERT"])

    # query
    user_question = st.text_input("Classify the sentence:")

    # add a button to run the model
    if st.button("Ask"):
        # run the model
        answer, score = predict_sentiment(user_question, app_mode)
        # print the answer and the score in red
        st.write(f"Sentiment: {answer}")
        st.write(f"Score: {score}")

    # load mcd dataset
    df = pd.read_csv('data/merged_df.csv', encoding='latin-1')
    df = df.loc[df['review'].str.contains(r'[^\x00-\x7F]+') == False]

    
    # location
    locations = df[['latitude ', 'longitude', 'store_address', 'rating']].dropna().drop_duplicates()
    # rename columns to match the format
    locations.columns = ['lat', 'lon', 'store_address', 'rating']

    print(locations.head())
    # create a folium map
   
    # Display the folium map using folium_static
    m = folium.Map()

    # Inside the loop where you add markers to the map
    for index, row in locations.iterrows():
        lat = row['lat']
        lon = row['lon']
        address = row['store_address']
        rating = np.round(row['rating'],2)
        positive = df.loc[df['store_address'] == address, 'positive'].iloc[0]
        negative = df.loc[df['store_address'] == address, 'negative'].iloc[0]
        neutral = df.loc[df['store_address'] == address, 'neutral'].iloc[0]
        tooltip = f"Address: {address}\nRating: {rating}"
        
        # Generate the pie chart HTML code for the specific store
        popup_html = generate_pie_chart(address, rating, positive, negative, neutral)
        
        # Create a popup with the pie chart HTML code
        popup = folium.Popup(popup_html, max_width=400)

        # Add a marker for the store with the popup
        folium.Marker(location=[lat, lon], tooltip=tooltip, popup=popup).add_to(m)

    # Display the folium map using folium_static
    st.markdown('### Map of McDonald\'s Stores')
    folium_static(m)
    # wordcloud
    # st.markdown('### Wordcloud')
    # plot_wordcloud(df['review'], title="Word Cloud of Reviews")
    

    # Show the reviews
    st.markdown('### Reviews')
    #st.write(df['review'])


if __name__ == '__main__':
    main()
