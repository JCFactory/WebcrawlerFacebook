from flask import Flask
from fapppack.reporting.mailer import Mailer
from fapppack.reporting.facebook_api import FacebookApi
from fapppack.tf_model.rnn_model import RnnModel
from pathlib import Path
from keras.preprocessing.text import Tokenizer



# Trainingsmodel

train_csv = Path("./measuring-customer-happiness/train_hp.csv")

test_csv = Path("./measuring-customer-happiness/test_hp.csv")
glove_file = Path('./glove.twitter.27B/glove.twitter.27B.100d.txt')
rnn_model = RnnModel(train_csv.absolute(), test_csv.absolute(), glove_file.absolute())

rnn_model.run()
# FBApi
fbapi = FacebookApi(rnn_model)
# Auswertung

# Mailversand



app = Flask(__name__)

@app.route('/')
def index():
    m = Mailer()
    return m.test()


if __name__ == '__main__':
    app.run(host="0.0.0.0")
