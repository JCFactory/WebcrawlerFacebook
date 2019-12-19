import json
import requests
import pandas as pd
import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class FacebookApi:

    ## below facebook api call
    model = None

    def __init__(self, webc_model):
        self.model = webc_model

    def callFacebookApi(self):
        url_long = "https://graph.facebook.com/v5.0/me?fields=id%2Cname%2Cposts%7Bcomments%7D&access_token=EAAlcIv35CUUBANREEygggKozZBFTNubNFhwuDn0u3MDI1jHz4PYRGirDFxz7MoSMTz3AHu6ZCQdT0oBp0cM3a30fGzw1pfb27C6H5OGn5jxiV49TqK8yaBiZA6DKZAeR3r0yzGGFFMGZAGmf6lMDeh8elMMGFZA5z4pk8KfQ2uiDZAZB67gt1nfh"
        response = requests.get(url_long)
        json_data = json.loads(response.text)
        dictFb = {'comment': [], 'time': []}

        for i in range(len(json_data['posts']['data'])):
            for j in range(len(json_data['posts']['data'][i]['comments']['data'])):
                message = self.model.preprocess_text(json_data['posts']['data'][i]['comments']['data'][j]['message'])
                dictFb['comment'].append(message)
                dictFb['time'].append(json_data['posts']['data'][i]['comments']['data'][j]['created_time'])

        df = pd.DataFrame(dictFb, columns=['comment', 'time'])
        return df

    def analyze(self):
        negcount = 0
        poscount = 0
        negseries = 0

        reportneg = False

        df_comments = self.callFacebookApi()
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


        df_comments.insert(2, "Sentiment", sentiment_series)
        sentiment_object = {'df_comments': df_comments, 'negative': negcount, 'positive': poscount, 'total': negcount + poscount,
                'reportnegative': reportneg,
                'negativepercent': negcount / (negcount + poscount) * 100,
                'positivepercent': poscount / (negcount + poscount) * 100}

        print(sentiment_object)

        return sentiment_object





