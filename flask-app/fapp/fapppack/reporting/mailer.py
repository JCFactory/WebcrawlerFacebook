import smtplib, ssl

class Mailer:

    def test(self):
        return "test2"

        # # Send email

    def sendMail(self):

        port = 587  # For starttls
        smtp_server = "smtp.gmail.com"
        receiver_email = "burghard.lachmann@gmail.com"
        sender_email = "HH.Analytica@gmail.com"
        password = 'WduenPa!19'
        message = """
        Report von HH Analytica
        """

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, port) as server:
            # server.ehlo()  # Can be omitted
            server.starttls(context=context)
            # server.ehlo()  # Can be omitted
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)

