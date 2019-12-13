import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase


class Mailer:
        # # Send email

    def sendMail(self, message=None):

        port = 587  # For starttls
        smtp_server = "smtp.gmail.com"
        receiver_email = "burghard.lachmann@gmail.com"
        sender_email = "HH.Analytica@gmail.com"
        password = 'WduenPa!f19'

        if(message == None):
            message = """
            Report von HH Analytica
            """
#
        try:
#             context = ssl.SSLContext(ssl.PROTOCOL_TLS)

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = 'Report von HH Analytica'

            body = message
            msg.attach(MIMEText(body,'plain'))

            message = msg.as_string()
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
            server.close()

            print ('Email sent!')
        except Exception as err:
            print('Exception')
            print(err)
#         context = ssl.create_default_context()
#         with smtplib.SMTP(smtp_server, port) as server:
#             # server.ehlo()  # Can be omitted
#             server.starttls(context=context)
#             # server.ehlo()  # Can be omitted
#             server.login(sender_email, password)
#             server.sendmail(sender_email, receiver_email, message)

