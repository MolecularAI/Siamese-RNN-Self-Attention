import sys 

class Unbuffered(object):
   """
   By default, print in Python is buffered, meaning that it does not write to files or stdout immediately, and needs to be 'flushed' to force the writing to stdout immediately.
   https://stackoverflow.com/questions/107705/disable-output-buffering
   
   """

   def __init__(self, stream):
       self.stream = stream

   def write(self, data):
       self.stream.write(data)
       self.stream.flush()

   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
       
   def __getattr__(self, attr):
       return getattr(self.stream, attr)