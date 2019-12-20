import pandas as pd

class DataStore:
    state = None
    time_reload = 60*60 # In seconds
    fb_api = None

    def __init__(self, fp_api):
        self.fb_api = fp_api

    def reload(self):
        self.fb_api
