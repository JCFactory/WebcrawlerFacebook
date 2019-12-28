import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt
from fpdf import FPDF
from datetime import date, datetime, timedelta
from pandas.plotting import register_matplotlib_converters


class Report:
    fakecsv = None
    report_data = None
    df = None
    static_folder = None

    def __init__(self, fakecsv, report, static_folder=''):
        self.fakecsv = fakecsv
        self.report_data = report
        self.static_folder = static_folder.as_posix()

    def generate_attachment(self):
        df = self.dataimport()
        self.weekplots(df)
        self.dayplot(df)
        return self.pdfgenerator(df)

    def execute_evaluation(self):
        summary = """ Report created:
            ## Total new Comments: {}
            ## Postive: {}
            ## Negative: {}
            ## Negative Percent: {}
            ## Positive Percent: {}"""

        print(self.report_data["df_comments"]["time"])
        df123 = self.report_data["df_comments"]
        csvfile = self.static_folder + "/export_dataframe.csv"
        df123.to_csv(csvfile, index = None, sep = '|')
        if len(self.report_data['newcomments']) > 0:
            summary += "\n\nNew Comments are:"
            for comment in self.report_data['newcomments']:
                summary += "\n\t"+comment

        return summary.format(self.report_data['total'], self.report_data['positive'], self.report_data['negative'],
                              self.report_data['negativepercent'], self.report_data['positivepercent'])

        # summary = """ Report erstellt:
        #     ## Gesamt Kommentare: """+self.report_data['total'] + """
        #     ## Postive: """+self.report_data['positive'] + """
        #     ## Negative: """+self.report_data['negative']
        #
        # return summary

    def set_dataframe(self, df_sent):
        self.df = df_sent

    def label(self, weekhour, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    # Variablen
    def execute_fakereport(self):
        return pdf

    def dataimport(self):
        # Create date parameter
        df = pd.DataFrame()

        # Read demo data into dataframe
        fakedata = np.loadtxt('FakeData.csv', delimiter=';')

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
        df['real_date'] = pd.TimedeltaIndex(df['date'], unit='d') + dt.datetime(1900, 1, 1)
        df['real_time'] = pd.to_timedelta(df["hour"], unit='h')
        df['real_datetime'] = pd.to_datetime(df['real_date'] + df['real_time'])
        # return (minus1, minus7, minus31, minus365)
        # return (count_minus1_neg, count_minus7_neg, count_minus31_neg, count_minus365_neg)
        # return (count_minus1_pos, count_minus7_pos, count_minus31_pos, count_minus365_neg)
        return df

        # Define and use a simple function to label the plot in axes coordinates

    def weekplots(self, df):
        # Prepares the dataframes for the weekplots
        df_week2 = df.copy()
        df_week2.drop(df_week2.columns.difference(['sentiment', 'weekhour', 'kw', 'real_datetime']), 1, inplace=True)
        kw_now = datetime.now().date().isocalendar()[1]
        df_week2.drop(df_week2[df_week2['kw'] < kw_now - 8].index, inplace=True)
        df_week2.drop(df_week2[df_week2['kw'] > kw_now - 1].index, inplace=True)
        df_week2["kw"] = df_week2["kw"].astype(int)
        df_week1 = df_week2.copy()
        df_week1.drop(df_week1[df_week1['sentiment'] == 1].index, inplace=True)
        df_week2.drop(df_week2.columns.difference(['weekhour', 'kw']), 1, inplace=True)
        df_week1.drop(df_week1.columns.difference(['weekhour', 'kw']), 1, inplace=True)

        # Creates the weekplot 1
        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        print(df_week1)
        weekplot1 = sns.FacetGrid(df_week1, row="kw", hue="kw", aspect=10, height=0.75, palette=pal)

        # Draw the densities in a few steps
        weekplot1.map(sns.kdeplot, "weekhour", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
        weekplot1.map(sns.kdeplot, "weekhour", clip_on=False, color="w", lw=2, bw=.2)
        weekplot1.map(plt.axhline, y=0, lw=2, clip_on=False)

        weekplot1.map(self.label, "weekhour")

        # Set the subplots to overlap
        weekplot1.fig.subplots_adjust(hspace=-0.5)

        # Remove axes details that don't play well with overlap
        weekplot1.set_titles("")
        weekplot1.set(yticks=[])
        weekplot1.axes[4, 0].set_ylabel('Kalenderwoche')
        weekplot1.axes[7, 0].set_xlabel('Wochenstunde 7x24')
        weekplot1.despine(bottom=True, left=True)

        weekplot1.savefig(self.static_folder + "/WeekPlot1.png")

        # Creates the weekplot 2
        weekplot2 = sns.FacetGrid(df_week2, row="kw", hue="kw", aspect=10, height=0.75, palette=pal)

        # Draw the densities in a few steps
        weekplot2.map(sns.kdeplot, "weekhour", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
        weekplot2.map(sns.kdeplot, "weekhour", clip_on=False, color="w", lw=2, bw=.2)
        weekplot2.map(plt.axhline, y=0, lw=2, clip_on=False)

        weekplot2.map(self.label, "weekhour")

        # Set the subplots to overlap
        weekplot2.fig.subplots_adjust(hspace=-0.5)

        # Remove axes details that don't play well with overlap
        weekplot2.set_titles("")
        weekplot2.set(yticks=[])
        weekplot2.axes[4, 0].set_ylabel('Kalenderwoche')
        weekplot2.axes[7, 0].set_xlabel('Wochenstunde 7x24')
        weekplot2.despine(bottom=True, left=True)

        weekplot2.savefig(self.static_folder + "/WeekPlot2.png")

    def dayplot(self, df):
        # Prepares the dataframe for the dayplot
        df_day = df.copy()
        df_day.drop(df_day.columns.difference(['sentiment', 'real_datetime']), 1, inplace=True)
        minus2 = datetime.today() - timedelta(days=2)
        now = datetime.now()
        df_day.drop(df_day[df_day['real_datetime'] < minus2].index, inplace=True)
        df_day.drop(df_day[df_day['real_datetime'] > now].index, inplace=True)
        df_day_sum = df_day.groupby(['real_datetime', 'sentiment']).size().unstack(fill_value=0)
        df_day_sum.columns = ['pos', 'neg']

        # Creates the dayplot
        plt.figure()
        sns.set(style="whitegrid")
        colors = ["#01355B", "#FFC000"]
        dayplot = sns.lineplot(data=df_day_sum, palette=(colors), linewidth=2.5)
        dayplot.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
        dayplot.set_ylabel('Anzahl')
        dayplot.set_xlabel('Uhrzeit')
        dayplotfigure = dayplot.get_figure()
        dayplotfigure.savefig(self.static_folder + "/DayPlot.png")

    def pdfgenerator(self, df):
        # Todays date
        today = date.today()
        strtoday = today.strftime("%d.%m.%Y")

        # variables
        minus1 = datetime.today() - timedelta(days=1)
        minus7 = datetime.today() - timedelta(days=7)
        minus31 = datetime.today() - timedelta(days=31)
        minus365 = datetime.today() - timedelta(days=365)

        # Counting different timeframes for the reporting tiles
        array_minus1_neg = df[(df.real_date > minus1) & (df.sentiment == 2)].count()
        array_minus7_neg = df[(df.real_date > minus7) & (df.sentiment == 2)].count()
        array_minus31_neg = df[(df.real_date > minus31) & (df.sentiment == 2)].count()
        array_minus365_neg = df[(df.real_date > minus365) & (df.sentiment == 2)].count()

        count_minus1_neg = array_minus1_neg[0]
        count_minus7_neg = array_minus7_neg[0]
        count_minus31_neg = array_minus31_neg[0]
        count_minus365_neg = array_minus365_neg[0]

        array_minus1_ges = df[(df.real_date > minus1)].count()
        array_minus7_ges = df[(df.real_date > minus7)].count()
        array_minus31_ges = df[(df.real_date > minus31)].count()
        array_minus365_ges = df[(df.real_date > minus365)].count()

        count_minus1_ges = array_minus1_ges[0]
        count_minus7_ges = array_minus7_ges[0]
        count_minus31_ges = array_minus31_ges[0]
        count_minus365_ges = array_minus365_ges[0]

        percent1 = round((count_minus1_neg / count_minus1_ges) * 100, 2)
        percent7 = round((count_minus7_neg / count_minus7_ges) * 100, 2)
        percent31 = round((count_minus31_neg / count_minus31_ges) * 100, 2)
        percent365 = round((count_minus365_neg / count_minus365_ges) * 100, 2)

        # Create strings for reporting tiles
        slash = '/'
        percent = '%'
        string_neg1 = str(count_minus1_neg)
        string_ges1 = str(count_minus1_ges)
        string1 = string_neg1 + slash + string_ges1
        string_neg7 = str(count_minus7_neg)
        string_ges7 = str(count_minus7_ges)
        string7 = string_neg7 + slash + string_ges7
        string_neg31 = str(count_minus31_neg)
        string_ges31 = str(count_minus31_ges)
        string31 = string_neg31 + slash + string_ges31
        string_neg365 = str(count_minus365_neg)
        string_ges365 = str(count_minus365_ges)
        string365 = string_neg365 + slash + string_ges365
        string_percent1 = str(percent1) + percent
        string_percent7 = str(percent7) + percent
        string_percent31 = str(percent31) + percent
        string_percent365 = str(percent365) + percent

        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_xy(0, 0)

        # Adding Layout
        pdf.image(self.static_folder + '/Report_Template.png', x=0, y=0, w=210, h=0, type='', link='')
        pdf.set_font('arial', '', 14)

        # Adding Plots
        pdf.set_xy(108, 111)
        pdf.image(self.static_folder + '/DayPlot.png', x=None, y=None, w=92, h=0, type='', link='')
        pdf.set_xy(15, 196)
        pdf.image(self.static_folder + '/WeekPlot1.png', x=None, y=None, w=85, h=0, type='', link='')
        pdf.set_xy(110, 196)
        pdf.image(self.static_folder + '/WeekPlot2.png', x=None, y=None, w=85, h=0, type='', link='')

        # Adding tile kpis
        pdf.set_xy(21, 132)
        pdf.cell(40, 10, str(string1))
        pdf.set_xy(67, 132)
        pdf.cell(40, 10, str(string7))
        pdf.set_xy(21, 167)
        pdf.cell(40, 10, str(string31))
        pdf.set_xy(67, 167)
        pdf.cell(40, 10, str(string365))
        pdf.set_xy(177, 75)
        pdf.cell(40, 10, strtoday)

        pdf.set_font('arial', '', 20)
        pdf.set_xy(21, 124)
        pdf.cell(40, 10, str(string_percent1))
        pdf.set_xy(67, 124)
        pdf.cell(40, 10, str(string_percent7))
        pdf.set_xy(21, 159)
        pdf.cell(40, 10, str(string_percent31))
        pdf.set_xy(67, 159)
        pdf.cell(40, 10, str(string_percent365))
        pdf.set_xy(177, 75)

        # Save PDF
        strtoday2 = today.strftime("%Y%m%d")
        filename = self.static_folder + '/Facebook_Report_' + str(strtoday2) + '.pdf'
        pdffile = pdf.output(filename, 'F')

        print('##### ', filename)
        return filename
