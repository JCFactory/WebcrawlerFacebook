import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class Mailer:
        # # Send email

    def sendMail(self, summary, attachment_pdf = None, attachment_csv=None):
        print('attachment_pdf ', attachment_pdf)
        subject = "An email with attachment from Python"
        body = "This is an email with attachment sent from Python"
        receiver_email = "burghard.lachmann@gmail.com"
        sender_email = "HH.Analytica@gmail.com"
        password = 'WduenPa!f19'

        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message["Bcc"] = receiver_email  # Recommended for mass emails

        # Add body to email
        message.attach(MIMEText(summary, "plain"))

#         filename = "PM-BigData-6 V1.pdf"  # In same directory as script
        # Open PDF file in binary mode
        if attachment_pdf:
            try:
                with open(attachment_pdf, "rb") as attachment:
                    # Add file as application/octet-stream
                    # Email client can usually download this automatically as attachment
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())

                # Encode file in ASCII characters to send by email
                    encoders.encode_base64(part)

                    # Add header as key/value pair to attachment part
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename=Facebook_Analytics_Report.pdf",
                    )

                    # Add attachment to message and convert message to string
                    message.attach(part)
            except IOError:
                print ("File not Found")

        if attachment_csv:
            try:
                with open(attachment_csv, "rb") as attachment:
                    # Add file as application/octet-stream
                    # Email client can usually download this automatically as attachment
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())

                # Encode file in ASCII characters to send by email
                    encoders.encode_base64(part)

                    # Add header as key/value pair to attachment part
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename=Facebook_Comments_Results.csv",
                    )

                    # Add attachment to message and convert message to string
                    message.attach(part)
            except IOError:
                print ("File not Found")


        text = message.as_string()

        # Log in to server using secure context and send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, text)


