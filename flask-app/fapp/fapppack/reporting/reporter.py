import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import timedelta
from fpdf import FPDF
from datetime import date


class Report:
    fakecsv = None
    report_data = None
    df = None

    def __init__(self, fakecsv, report=None):
        self.fakecsv = fakecsv
        self.report_data = report

    def execute_evaluation(self):
        summary = """ Report created:
            ## Total new Comments: {}
            ## Postive: {}
            ## Negative: {}
            ## Negative Percent: {}
            ## Positive Percent: {}"""

        return summary.format(self.report_data['total'], self.report_data['positive'], self.report_data['negative'],
                              self.report_data['negativepercent'], self.report_data['positivepercent'])

        # summary = """ Report erstellt:
        #     ## Gesamt Kommentare: """+self.report_data['total'] + """
        #     ## Postive: """+self.report_data['positive'] + """
        #     ## Negative: """+self.report_data['negative']
        #
        # return summary

    def set_dataframe(self, df_sent):
        self.df = df

    # Variablen
    def execute_fakereport(self):
        count_minus1_neg = 0
        count_minus7_neg = 0
        count_minus31_neg = 0
        count_minus365_neg = 0
        count_minus1_pos = 0
        count_minus7_pos = 0
        count_minus31_pos = 0
        count_minus365_pos = 0
        now = datetime.now()
        minus1 = datetime.today() - timedelta(days=1)
        minus7 = datetime.today() - timedelta(days=7)
        minus31 = datetime.today() - timedelta(days=31)
        minus365 = datetime.today() - timedelta(days=365)

        fakedata = np.loadtxt(self.fakecsv, delimiter=';')

        df = pd.DataFrame()
        df['id'] = fakedata[:, 0]
        df['date'] = fakedata[:, 1]
        df['month'] = fakedata[:, 2]
        df['kw'] = fakedata[:, 3]
        df['weekday'] = fakedata[:, 4]
        df['day'] = fakedata[:, 7]
        df['hour'] = fakedata[:, 8]
        df['minute'] = fakedata[:, 9]
        df['weekhour'] = fakedata[:, 10]
        df['sentiment'] = fakedata[:, 11]
        df['real_date'] = pd.TimedeltaIndex(df['date'], unit='d') + datetime.datetime(1900, 1, 1)

        array_minus1_neg = df[(df.real_date > minus1) & (df.sentiment == 2)].count()
        array_minus7_neg = df[(df.real_date > minus7) & (df.sentiment == 2)].count()
        array_minus31_neg = df[(df.real_date > minus31) & (df.sentiment == 2)].count()
        array_minus365_neg = df[(df.real_date > minus365) & (df.sentiment == 2)].count()

        count_minus1_neg = array_minus1_neg[0]
        count_minus7_neg = array_minus7_neg[0]
        count_minus31_neg = array_minus31_neg[0]
        count_minus365_neg = array_minus365_neg[0]

        array_minus1_pos = df[(df.real_date > minus1) & (df.sentiment == 1)].count()
        array_minus7_pos = df[(df.real_date > minus7) & (df.sentiment == 1)].count()
        array_minus31_pos = df[(df.real_date > minus31) & (df.sentiment == 1)].count()
        array_minus365_pos = df[(df.real_date > minus365) & (df.sentiment == 1)].count()

        count_minus1_pos = array_minus1_pos[0]
        count_minus7_pos = array_minus7_pos[0]
        count_minus31_pos = array_minus31_pos[0]
        count_minus365_pos = array_minus365_pos[0]

        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # Create the data
        rs = np.random.RandomState(1979)
        x = rs.randn(500)
        g = np.tile(list("ABCDEFGHIJ"), 50)
        df = pd.DataFrame(dict(x=x, g=g))
        m = df.g.map(ord)
        df["x"] += m

        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(dfcsv, row="g", hue="g", aspect=15, height=.5, palette=pal)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
        g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)

        g.map(label, "x")

        # Set the subplots to overlap
        g.fig.subplots_adjust(hspace=-.25)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)

        g.savefig("WeekPlot.png")

        sns.set(style="whitegrid")

        # rs = np.random.RandomState(365)
        # values = rs.randn(365, 4).cumsum(axis=0)
        # dates = pd.date_range("1 1 2016", periods=365, freq="D")
        values = df[(df.real_date > minus1)].count()
        print(values)
        hour = df.hour
        print(hour)
        data = pd.DataFrame(values, hour, columns=["Positiv", "Negativ", "Gesant"])
        data = data.rolling(7).mean()

        dayplot = sns.lineplot(data=data, palette="tab10", linewidth=2.5)
        print(dayplot)
        dayplotfigure = dayplot.get_figure()
        dayplotfigure.savefig("DayPlot.png")

        today = date.today()
        strtoday = today.strftime("%d.%m.%Y")

        slash = '/'
        string_neg1 = str(count_minus1_neg)
        string_pos1 = str(count_minus1_pos)
        string1 = string_neg1 + slash + string_pos1
        string_neg7 = str(count_minus7_neg)
        string_pos7 = str(count_minus7_pos)
        string7 = string_neg7 + slash + string_pos7
        string_neg31 = str(count_minus31_neg)
        string_pos31 = str(count_minus31_pos)
        string31 = string_neg31 + slash + string_pos31
        string_neg365 = str(count_minus365_neg)
        string_pos365 = str(count_minus365_pos)
        string365 = string_neg365 + slash + string_pos365

        pdf = FPDF()
        pdf.add_page()
        pdf.set_xy(0, 0)
        pdf.image('Report_Template.png', x=0, y=0, w=210, h=0, type='', link='')
        pdf.set_font('arial', '', 14)
        pdf.set_xy(108, 111)
        pdf.image('DayPlot.png', x=None, y=None, w=92, h=0, type='', link='')
        pdf.set_xy(19, 198)
        pdf.image('WeekPlot.png', x=None, y=None, w=81, h=0, type='', link='')
        pdf.set_xy(110, 198)
        pdf.image('WeekPlot2.png', x=None, y=None, w=81, h=0, type='', link='')
        pdf.set_xy(21, 127)
        pdf.cell(40, 10, str(string1))
        pdf.set_xy(67, 127)
        pdf.cell(40, 10, str(string7))
        pdf.set_xy(21, 161)
        pdf.cell(40, 10, str(string31))
        pdf.set_xy(67, 161)
        pdf.cell(40, 10, str(string365))
        pdf.set_xy(177, 75)
        pdf.cell(40, 10, strtoday)
        pdf = pdf.output('HamburgAnalytica.pdf', 'F')

        return pdf



