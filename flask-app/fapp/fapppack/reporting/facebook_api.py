import json
import requests
import pandas as pd
import datetime


class FacebookApi:

    ## below facebook api call

    def callFacebookApi(self):
        url_long = "https://graph.facebook.com/v5.0/me?fields=id%2Cname%2Cposts%7Bcomments%7D&access_token=EAAlcIv35CUUBANREEygggKozZBFTNubNFhwuDn0u3MDI1jHz4PYRGirDFxz7MoSMTz3AHu6ZCQdT0oBp0cM3a30fGzw1pfb27C6H5OGn5jxiV49TqK8yaBiZA6DKZAeR3r0yzGGFFMGZAGmf6lMDeh8elMMGFZA5z4pk8KfQ2uiDZAZB67gt1nfh"

        response = requests.get(url_long)
        json_data = json.loads(response.text)
        # df = pd.DataFrame(columns=['comment', 'person', 'time'])
        dictFb = {'comment': [], 'person': [], 'time': []}
        for i in range(len(json_data['posts']['data']) - 1):
            dictFb['comment'].append(json_data['posts']['data'][i]['comments']['data'][i]['message'])
            dictFb['person'].append(json_data['posts']['data'][i]['comments']['data'][i]['from']['name'])
            dictFb['time'].append(json_data['posts']['data'][i]['comments']['data'][i]['created_time'])
            # df.append({'comment': comment, 'person':person, 'time':time}, ignore_index=True)
        df = pd.DataFrame(dictFb, columns=['comment', 'person', 'time'])

        return df
