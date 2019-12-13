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
                dictFb['comment'].append(json_data['posts']['data'][i]['comments']['data'][j]['message'])
                dictFb['time'].append(json_data['posts']['data'][i]['comments']['data'][j]['created_time'])

        df = pd.DataFrame(dictFb, columns=['comment', 'time'])
        return df

    # def analyze(self):
    #     df_comments = self.callFacebookApi()
    #     tokenizer = Tokenizer(num_words=5000)
    #     for el in df_comments['posts']:
    #         el = tokenizer.texts_to_sequences(el)
    #         flat_list = []
    #         for sublist in el:
    #             for item in sublist:
    #                 flat_list.append(item)
    #         flat_list = [flat_list]
    #         el = pad_sequences(flat_list, padding='post', maxlen=100)
    #
    #         predictvalue = self.model.predict(el)
    #
    #         if (predictvalue < 0.5):
    #             sentiment = 'negativ'
    #             df_comments.insert(2, "Sentiment", sentiment, True)
    #         elif (predictvalue > 0.5):
    #             sentiment = 'positiv'
    #             df_comments.insert(2, "Sentiment", sentiment, True)
    #     print(df_comments)
    #     return df_comments


    def analyze(self):
        negcount = 0
        poscount = 0
        negseries = 0

        reportneg = False

        df_comments = self.callFacebookApi()

        tokenizer = Tokenizer(num_words=5000)
        for element in df_comments['comment']:
            print(element)
            print(type(element))
            tokenizer.fit_on_texts(texts=element)
            el = tokenizer.texts_to_sequences(texts=list(element))
            flat_list = []
            for sublist in el:
                for item in sublist:
                    flat_list.append(item)
            flat_list = [flat_list]
            el = pad_sequences(flat_list, padding='post', maxlen=100)

            predictvalue = self.model.predict(el)

            if (predictvalue <= 0.5):
                negcount += 1
                negseries += 1
                sentiment = 'negativ'
                df_comments.insert(2, "Sentiment", sentiment, True)

            elif (predictvalue > 0.5):
                poscount += 1
                negseries = 0
                sentiment = 'positiv'
                df_comments.insert(2, "Sentiment", sentiment, True)

            if (negseries == 5 and reportneg == False):
                reportneg = True

        sentiment_object = {'df_comments': df_comments, 'negative': negcount, 'positive': poscount, 'total': negcount + poscount,
                'reportnegative': reportneg,
                'negativepercent': negcount / (negcount + poscount) * 100,
                'positivepercent': poscount / (negcount + poscount) * 100}

        print(sentiment_object)

        return sentiment_object





