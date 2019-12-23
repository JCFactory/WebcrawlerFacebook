from requests import request
import time
# from daemon import runner
import os

import threading


RELOAD_FB = os.environ.get('RELOAD_FB')
if RELOAD_FB:
    RELOAD_FB = int(RELOAD_FB)
else:
    RELOAD_FB = 10 #60 * 60  # Repeat every hour if nothing set


def generate_reports():
    print('Starting of thread :', threading.currentThread().name)
    time.sleep(RELOAD_FB)
    req = request('GET', 'http://localhost:8080/report')
    print(req.content)

    print('Finishing of thread :', threading.currentThread().name)
    generate_reports()


a = threading.Thread(target=generate_reports, name='Thread-a', daemon=True)
a.start()
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
