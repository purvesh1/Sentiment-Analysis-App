import tweepy
import twitter_credentials
import configparser
import pandas as pd

# read configs
config = configparser.ConfigParser()
config.read('config.ini')

# api_key = config['twitter']['api_key']
# api_key_secret = config['twitter']['api_key_secret']

# access_token = config['twitter']['access_token']
# access_token_secret = config['twitter']['access_token_secret']

# bearer_token = config['twitter']['bearer_token']

# authentication
auth = tweepy.OAuthHandler(twitter_credentials.API_KEY, twitter_credentials.API_KEY_SECRET)
auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# get tweets about a hashtag or a keyword
client = tweepy.Client(bearer_token=twitter_credentials.BEARER_TOKEN)

# Pull tweets from twitter
query = '#trump -is:retweet lang:en'
tweets = client.search_recent_tweets(query=query, max_results=2)

# Get tweets that contain the hashtag #TypeKeywordHere
# -is:retweet means I don't want retweets
# lang:en is asking for the tweets to be in english
# print pulled tweets
for tweet in tweets.data:
    print('\n**Tweet Text**\n',tweet.text)
    

# create dataframe
columns = ['Time', 'User', 'Tweet']
data = []
for tweet in tweets.data:
    data.append([tweet.created_at, tweet.text])

df = pd.DataFrame(data, columns=columns)
df.to_csv('tweets.csv')