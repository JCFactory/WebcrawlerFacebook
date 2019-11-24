from flask import Flask

from reporting.mailer import Mailer
app = Flask(__name__)


@app.route('/')
def index():
    m = Mailer()
    return m.test()


if __name__ == '__main__':
    app.run(host="0.0.0.0")
