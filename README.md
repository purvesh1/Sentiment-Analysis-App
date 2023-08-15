# Sentiment-Analysis-App
I/p a sentence, o/p the classified sentiment


### Plan of action:
1. Create a modular template and include  anything else if required
2. Create a virtual environment
3. Update and run setup.py
4. Install requirements.txt
5. Test and save the model (research/semeval: save_pretrained)
6. Create a streamlit app
7. [Checkpoint 1] Load the pretained model and tokenizer and use them to predict the sentiment per aspect of the input sentence
8. [Checkpoint 2] Load a dataset with reviews and locations, converrt to aspect based format, predict the sentment per aspect and plot the avg sentiment per location for each aspect
9. Use Google Maps API to periodically get this data and periodically run the model.
10. Save a update the data on a database
11. Deploy the app 
