import json
import requests
class FacebookApi:

    def test(self):
        return "test"

    # api-endpoint
    burghard_api_url = "https://graph.facebook.com/v4.0/me?fields=id%2Cname%2Cposts%7Bcomments%2Clikes%2Creactions%7Bname%2Ctype%2Cid%7D%7D%2Cfeed&access_token=EAAlcIv35CUUBACAL7BAOaT1Fm4mn84wsn0yqOr2UU9R5FG59beXoymQAdA47G4jQqtcW9iZBhCrwlW0VxOQNEUTNgWK4ZBM6vjTA9BthBw7jwM55RKKVgYRt0V4tFcQZAthLHPMZBAZBZBnoHRpD3PXeZBZBilgnaxHJuHoMg7E4dpYdhZAbioeaA33hviyR6DZAj6ZBWh3896phZAo8ft9yV26Y"

    lachmann_cruises_api_url = "https://graph.facebook.com/v5.0/me?fields=id%2Cname%2Cposts%7Bcomments%2Csharedposts%7D&access_token=EAAlcIv35CUUBAKDaXZA3zrldcprNX4RosQpbZBAVn5WvMmzp6fwh3Q17zB6qiHMf3OZCiLdx3GSrhGma0fDkYwPGQzj6zYqQw3pdpv5ZBAUoWZAsO2OiCUKHlBvMNOp2lDx2On9q0jWZC7pC2XynX6OigHn9TswWQfpfaxGS2tIRYp1Dm2JYEHTKAlbf5juQIkoXmnKqk1wsIDFUqYebjj"

    new_url = "https://graph.facebook.com/v5.0/me?fields=id%2Cname%2Clikes%2Cposts&access_token=EAAlcIv35CUUBAEXpcPy1a1OtHIrEw3qD79EZBZCk0oAPBYO8F7tvg5xKvV3XcbevYdgwvjE9vTXVAc4cW014I5FkGlpIQgjAZAi8c0C0cIJSNqYzqJfP78Bq1Cdc6u8mITjpdRZCx73dsrjYyfQsLiKRZCdAShWSpHrFHj7CXbvim8hYSjFbwWreIt2mOUZAmATaXqTHNUhKdXsO9cNHxS"

    response = requests.get(new_url)
    json_data = json.loads(response.text)
    posts = json_data['posts']

    for i in posts['data']:
        message = i['message']
        print(message)
