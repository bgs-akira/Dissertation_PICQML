import numpy as np
import serial  # Import the serial library
from time import sleep
from abc import ABC, abstractmethod


class OpticalSwitch(ABC):

    @abstractmethod
    def switch(self, channel):
        pass

    @abstractmethod
    def reset(self):
        pass


class LF30CHSM(OpticalSwitch):

    def __init__(self, resource):
        '''
        :param resource: e.g. 'COM4' or ['COM4]
        '''

        resource = np.atleast_1d(resource)
        resource = resource[0]

        self.SerialData = serial.Serial(resource, baudrate=115200, timeout=3.0, writeTimeout=0)

        self.num_channels = 30
        self.wait_time = 1  # wait time after switching channels

        self.reset()  # reset upon init

    def _serial_wait(self):
        # runs danger of an infinite loop, hence make private.
        sleep(0.1)
        # Remove cached readout.
        while True:
            if (self.SerialData.inWaiting() > 0):
                myData = self.SerialData.readline()
                break
        sleep(self.wait_time)

    def reset(self):
        self.SerialData.write(b'r\n')
        self._serial_wait()

    def switch(self, channel):
        if channel < 1 or channel > self.num_channels:
            raise ValueError('Channel out of range')

        channel = channel - 1
        ch_b = f'{channel:05b}'
        ch_b = ch_b[::-1] + '0'
        command = f's{ch_b}'
        self.SerialData.write(command.encode())
        self._serial_wait()

    def query(self):
        self.SerialData.write(b'?')
        self._serial_wait()

        while True:
            if (self.SerialData.inWaiting() > 0):
                myData = self.SerialData.readline().decode('utf-8')
                break

        return myData

    def close(self):
        self.SerialData.close()


class Hand(OpticalSwitch):
    def __init__(self):
        print('The optical switch is your hand. ')

    def switch(self, channel):
        s = False
        while not s:
            x = input(f'Have you switched to channel {channel}? [Y/n]')
            s = (x == 'Y')
    def reset(self):
        print('Rest your hand now.')


if __name__ == "__main__":
    o_switch = LF30CHSM('COM4')
