import json
import requests
import pandas as pd
import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from fapppack.tf_model.data_store import DataStore


class FacebookApi:

    ## below facebook api call
    model = None
    new_messages = []

    def __init__(self, webc_model):
        self.model = webc_model

    def callFacebookApi(self, incremental=False):
        url_long = "https://graph.facebook.com/v5.0/me?fields=id%2Cname%2Cposts%7Bcomments%7D&access_token=EAAlcIv35CUUBANREEygggKozZBFTNubNFhwuDn0u3MDI1jHz4PYRGirDFxz7MoSMTz3AHu6ZCQdT0oBp0cM3a30fGzw1pfb27C6H5OGn5jxiV49TqK8yaBiZA6DKZAeR3r0yzGGFFMGZAGmf6lMDeh8elMMGFZA5z4pk8KfQ2uiDZAZB67gt1nfh"
        response = requests.get(url_long)
        json_data = json.loads(response.text)
        # print(json_data)

        if incremental == False:
            DataStore.dataset = dict()
        dictFb = {'comment': [], 'time': [], 'original_message': []}
        x = 0
        for i in range(len(json_data['posts']['data'])):
            for j in range(len(json_data['posts']['data'][i]['comments']['data'])):
                message_data = json_data['posts']['data'][i]['comments']['data'][j]
                message_id = message_data['id']
                original_message = message_data['message']
                message = self.model.preprocess_text(original_message)
                if not DataStore.dataset.__contains__(message_id):
                    DataStore.state = True
                    self.new_messages.append(message_data['message'])
                dictFb['original_message'].append(original_message)
                dictFb['comment'].append(message)
                dictFb['time'].append(message_data['created_time'])
                DataStore.dataset[message_id] = message_data
        # return
        if DataStore.state:
            df = pd.DataFrame(dictFb, columns=['original_message', 'comment', 'time'])
            DataStore.state = False
            return df
        else:
            return None

    def analyze(self, incremental=False):
        negcount = 0
        poscount = 0
        negseries = 0

        reportneg = False

        df_comments = self.callFacebookApi(incremental=incremental)
        if (df_comments is None) or df_comments.empty:
            return None
        sentiment_series = []
        for element in df_comments['comment']:
            predictvalue = self.model.predict(element)
            predictvalue = predictvalue[0]

            if (predictvalue <= 0.5):
                negcount += 1
                negseries += 1
                sentiment = 'negativ'
                sentiment_series.append(sentiment)

            elif (predictvalue > 0.5):
                poscount += 1
                negseries = 0
                sentiment = 'positiv'
                sentiment_series.append(sentiment)
            if (negseries == 5 and reportneg == False):
                reportneg = True


        df_comments.insert(3, "Sentiment", sentiment_series)
        sentiment_object = {'df_comments': df_comments, 'negative': negcount, 'positive': poscount, 'total': negcount + poscount,
                'reportnegative': reportneg,
                'negativepercent': negcount / (negcount + poscount) * 100,
                'positivepercent': poscount / (negcount + poscount) * 100,
                'newcomments': self.new_messages}

        return sentiment_object





