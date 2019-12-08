from flask import Flask

from fapppack.reporting.mailer import Mailer
from fapppack.tf_model.rnn_model import RnnModel
app2 = Flask(__name__)

# Trainingsmodel
# FBApi
# Auswertung
# Mailversand



@app2.route('/')
def index():
    m = Mailer()
    return m.test()


if __name__ == '__main__':
    app2.run(host="0.0.0.0")
