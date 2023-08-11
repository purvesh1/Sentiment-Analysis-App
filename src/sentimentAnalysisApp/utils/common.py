import os
from box.exceptions import BoxValueError
import yaml
from sentimentAnalysisApp.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import folium
from streamlit_folium import folium_static
import numpy as np
import matplotlib.pyplot as plt
import base64
import io

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

@ensure_annotations
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

@ensure_annotations
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

@ensure_annotations
def get_address(lat, lon):
    geolocator = Nominatim(user_agent="sentiment_analysis_app")
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        return location.address
    except GeocoderTimedOut:
        return get_address(lat, lon)


    # Load the tokenizer and model (trainer)
    tokenizer = load_tokenizer('path_to_tokenizer')
    trainer = load_model('path_to_model')

    st.title('Aspect Based Sentiment Analysis')

    # Textbox for user input
    user_input = st.text_area("Enter a review:")

    # Predict button
    if st.button('Predict'):
        predicted_labels, scores = predict_sentiment(user_input, tokenizer, trainer)
        st.write(f"Predicted Label: {predicted_labels[0]}")  # Display the first label for demonstration
        # You can further process and display results as needed