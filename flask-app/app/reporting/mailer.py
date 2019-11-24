import smtplib, ssl

class Mailer:

    def test(self):
        return "test"

        # # Send email


    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    sender_email = "burghard.lachmann@gmail.com"
    receiver_email = "jackyaudrey2015@gmail.com"
    password = 'Brunhilde'
    message = """Subject: Hi there

       This message is sent from Python."""

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        # server.ehlo()  # Can be omitted
        server.starttls(context=context)
        # server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)



