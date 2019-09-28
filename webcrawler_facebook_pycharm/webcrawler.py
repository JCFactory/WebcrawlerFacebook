# importing the requests library
import requests

# api-endpoint
api_url = "https://graph.facebook.com/v4.0/me?fields=id%2Cname%2Cposts%7Bcomments%2Clikes%2Creactions%7Bname%2Ctype%2Cid%7D%7D%2Cfeed&access_token=EAAlcIv35CUUBACAL7BAOaT1Fm4mn84wsn0yqOr2UU9R5FG59beXoymQAdA47G4jQqtcW9iZBhCrwlW0VxOQNEUTNgWK4ZBM6vjTA9BthBw7jwM55RKKVgYRt0V4tFcQZAthLHPMZBAZBZBnoHRpD3PXeZBZBilgnaxHJuHoMg7E4dpYdhZAbioeaA33hviyR6DZAj6ZBWh3896phZAo8ft9yV26Y"

# sending get request and saving the response as response object
r = requests.get(url=api_url)

