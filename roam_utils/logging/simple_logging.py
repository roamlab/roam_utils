import os

class Logger(object):
    def __init__(self, logfile):
        if os.path.exists(logfile):
            # if os.path.getsize(logfile) > 0:
            #     print(logfile + ' is not empty. Creating new file.')
            #     self.logfile = logfile + '1'
            # else:
            #     self.logfile = logfile
            print(logfile + ' is not empty. Overwritten!')

        self.logfile = logfile
        open(self.logfile,'w').close()

    def log_info(self, msg_str):
        with open(self.logfile, 'a') as f:
            f.write(msg_str)
            f.write('\n')


