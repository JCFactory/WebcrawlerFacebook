from flask import Flask
import tensorflow
from tensorflow import keras as k

from .reporting.mailer import Mailer
app = Flask(__name__)


@app.route('/')
def hello_world():
    m = Mailer()
    return m.test()


if __name__ == '__main__':
    app.run(host="0.0.0.0")
