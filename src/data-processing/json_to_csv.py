import json_lines
import sys

def load_jsonl(file):
    tweets = []
    with open(file, 'rb') as f:
        for tweet in json_lines.reader(f, broken=True):
            tweets.append(tweet)
    return (tweets)


json_file_name = sys.argv[1]
csv_file_name = sys.argv[2]

tweets = load_jsonl(json_file_name)

outtweets = []
for tweet in tweets:
    tw = [tweet['created_at'] 
            , tweet['id_str']
            , tweet['user']['id_str']
            , tweet['user']['name']
            , tweet['user']['screen_name']
            , tweet['user']['description']
            , tweet['text']
            #, tweet['user']['derived']['locations']['country']
            #, tweet['user']['derived']['locations']['country_code']
            #, tweet['user']['derived']['locations']['locality']
            #, tweet['user']['derived']['locations']['region']
            , tweet['user']['followers_count']
            , tweet['user']['friends_count']
            , tweet['user']['listed_count']
            , tweet['user']['favourites_count']
            , tweet['user']['statuses_count']
            , tweet['user']['created_at']
            , tweet['retweeted']
            , tweet['retweet_count']
         ]
    location_info = ['na','na','na']
    if 'derived' in tweet['user']:
        if 'locations' in tweet['user']['derived']:
            if 'region' in tweet['user']['derived']['locations'][0]:
                location_info = [tweet['user']['derived']['locations'][0]['country']
                                 , tweet['user']['derived']['locations'][0]['country_code']
                                 #, tweet['user']['derived']['locations'][0]['locality']
                                 , tweet['user']['derived']['locations'][0]['region']
                                ]
    tw.extend(location_info)

    full_text = 'na'
    if 'extended_tweet' in tweet:
        if 'full_text' in tweet['extended_tweet']:
            full_text = tweet['extended_tweet']['full_text']
    tw.append(full_text)
        
    outtweets.append(tw)
    




import pandas as pd
df_tweets  = pd.DataFrame(outtweets
                          , columns =['ts', 'id_str','user_id','user_name','screen_name', 'description', \
                                      'text', 'followers_count', 'friends_count', 'listed_count', \
                                      'favourites_count', 'statuses_count','profile_created', \
                                      'retweeted','retweet_count', 'country','country_code','region', 'full_text'
                                      ]
                          )


print(df_tweets.head(5))

print('number of tweets processed: ' + str(df_tweets.shape))

df_tweets.to_csv(csv_file_name, index = False)





