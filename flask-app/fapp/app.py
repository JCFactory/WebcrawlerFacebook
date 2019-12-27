from flask import Flask
from fapppack.reporting.mailer import Mailer
from fapppack.reporting.facebook_api import FacebookApi
from fapppack.tf_model.training_model import RnnModel
from fapppack.reporting.report import Report
from pathlib import Path
from keras.preprocessing.text import Tokenizer
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Trainingsmodel

static_folder = Path("./static/")
train_csv = Path("./measuring-customer-happiness/train_hp.csv")
fakecsv = Path("./FakeData.csv")
test_csv = Path("./measuring-customer-happiness/test_hp.csv")
glove_file = Path('./glove/glove.twitter.27B.50d.txt')
glove_files = [
    Path('./glove/glove.twitter.27B.50d.txt_00'),
    Path('./glove/glove.twitter.27B.50d.txt_01'),
    Path('./glove/glove.twitter.27B.50d.txt_02'),
    Path('./glove/glove.twitter.27B.50d.txt_03'),
    Path('./glove/glove.twitter.27B.50d.txt_04'),
    Path('./glove/glove.twitter.27B.50d.txt_05')
]
rnn_model = RnnModel(train_csv.absolute(), test_csv.absolute(), glove_file.absolute(), glove_files=glove_files)
rnn_model.run()


# FBApi

def execute_reports(incremental=False):
    print('Start Rep')
    fbapi = FacebookApi(rnn_model)
    # Auswertung
    data = fbapi.analyze(incremental=incremental)
    if data is not None:
        rep = Report(fakecsv.absolute(), data, static_folder=static_folder.absolute())
        summary = rep.execute_evaluation()
        pdf = rep.generate_attachment()
        # Mailversand

        print(pdf)
        mail = Mailer()
        mail.sendMail(summary=summary, attachment_pdf=pdf)
        print('End Rep')
        return summary
    else:
        return "No Data"


execute_reports()

app = Flask(__name__)


@app.route('/')
def index():
    return "WELCOME!!!!"


@app.route('/report')
def report():
    return execute_reports(True)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
