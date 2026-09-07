import pyvisa
import time

class PM100D(object):
    def __init__(self, resource, wavelength=1550):

        rm = pyvisa.ResourceManager()
        self.inst = rm.open_resource(resource, timeout=1)
        self.inst.read_termination = '\n'
        self.inst.write_termination = '\n'
        self.inst.timeout = 1000

        self.set_wav(wavelength)

    def set_wav(self, wav):
        self.inst.write(f'CORR:WAV {wav:4d}')  # set wavelength

    def read(self):
        return float(self.inst.query('MEAS:POW?'))

