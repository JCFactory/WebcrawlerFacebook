from requests import request
import time
# from daemon import runner
import os
import threading
from flask import Flask
import sys


RELOAD_FB = os.environ.get('RELOAD_FB')
REQUEST_SERVICE = os.environ.get('REQUEST_SERVICE')
if RELOAD_FB:
    RELOAD_FB = int(RELOAD_FB)
else:
    RELOAD_FB = 10  # 60 * 60  # Repeat every hour if nothing set

print("Relaod Time")
print(RELOAD_FB)


def generate_reports():
        try:
            print('Starting of thread :', threading.currentThread().name)
            url = 'http://' + REQUEST_SERVICE + ':8080/report'
            print(url)
            req = request('GET', url)
            print(req.content)
            print('Sleeping for ', RELOAD_FB, ' Sek')
        except:
            print ("Unexpected error:", sys.exc_info()[0])
        time.sleep(RELOAD_FB)
        print('Finishing of thread :', threading.currentThread().name, flush=True)
        generate_reports()


app = Flask(__name__)


@app.route('/')
def index():
    return "WELCOME!!!!"

# generate_reports()

if __name__ == '__main__':
    a = threading.Thread(target=generate_reports, name='Thread-Daemon', daemon=True)
    a.start()
    app.run(host="0.0.0.0", port=8082)
    # generate_reports()
# class App():
#     def __init__(self):
#         self.stdin_path = '/dev/null'
#         self.stdout_path = '/dev/tty'
#         self.stderr_path = '/dev/tty'
#         self.pidfile_path =  '/tmp/foo.pid'
#         self.pidfile_timeout = 5
#     def run(self):
#         while True:
#             time.sleep(RELOAD_FB)
#             request('GET', 'http://localhost:8080/report/')
#
# app = App()
# daemon_runner = runner.DaemonRunner(app)
# daemon_runner.do_action()
