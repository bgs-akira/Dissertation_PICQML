import numpy as np
import serial  # Import the serial library
# import logging
from time import sleep
from abc import ABC, abstractmethod

# TODO: set argument names to v and c. SIMPLER!
class PowerSupply(ABC):

    @abstractmethod
    def set_voltage(self, channel, v):
        pass

    @abstractmethod
    def set_current(self, channel, c):
         pass

    @abstractmethod
    def set_channel(self, channel, v, c):
        pass

    @abstractmethod
    def read_channel(self, channel):
        pass

    @abstractmethod
    def read_voltage(self, channel):
        pass

    @abstractmethod
    def read_current(self, channel):
        pass

    @abstractmethod
    def zero_all(self):
        pass

    @abstractmethod
    def close(self):
        pass


class XPOW(PowerSupply):

    def __init__(self, resource):
        '''
        :param resource: e.g. ['COM8', 'COM9', 'COM10']
        '''

        # write these into properties and make them immutable
        # ABSOLUTELY DO NOT CHANGE THESE PARAMETERS
        self._safe_voltage = 20
        self._safe_current = 100

        # Currently there is only one safe mode of operation, and that is keep voltage fixed and sweep current.
        self._current_step = 5  # current step increase should not exceed this
        self._voltage_step = 0.  # voltage should be kept fixed

        # DO NOT CHANGE THESE PARAMETERS
        self.line_num = 3
        self.channel_num = 40

        if len(resource) != self.line_num:
            raise ValueError('There should be three lines in total')

        # Connect device
        SerialDatas = []
        for com_channel in resource:
            # Search device through VID:PID for serial communication and stop the setting with last setting is bank
            # values 3
            SerialDatas.append(serial.Serial(com_channel, baudrate=115200, timeout=3.0, writeTimeout=0))
        self.SerialDatas = SerialDatas

        # Memory cache of set values, not actual values
        self._cache_current = [0] * 120
        self._cache_voltage = [0] * 120

    @property
    def safe_voltage(self):
        return self._safe_voltage

    @property
    def safe_current(self):
        return self._safe_current

    @property
    def current_step(self):
        return self._current_step

    @property
    def voltage_step(self):
        return self._voltage_step

    def get_cache_current(self, channel):
        return self._cache_current[channel - 1]

    def get_cache_voltage(self, channel):
        return self._cache_voltage[channel - 1]

    @staticmethod
    def calc_line_ind(channel):
        ''' Calculate which line (0,1,2) and the corresponding channel index in that line does the channel (1-120)
        belong'''

        if 1 <= channel <= 40:
            line_ind = 0
        elif 41 <= channel <= 80:
            line_ind = 1
        elif 81 <= channel <= 120:
            line_ind = 2
        else:
            raise ValueError(f'channel_num not in range [1, 120]')

        return line_ind

    @staticmethod
    def convert_voltage(v):
        max_voltage = 29

        # # nicslab version
        # if 0.1 <= v < 12:
        #     scaled = v + 0.5
        # elif 12 <= v < 17.1:
        #     scaled = v + 0.3
        # elif 0 <= v <= max_voltage:
        #     scaled = v

        # inhouse version
        if 0.1 <= v < 12:
            scaled = v + 1 + 0.025 * (v - 1)
        elif 12 <= v < 17.1:
            scaled = v + 1.3
        elif 0 <= v <= max_voltage:
            scaled = v
        else:
            raise ValueError(f'Voltage outside range [0, {max_voltage}]')

        return 65535 * scaled / max_voltage

    @staticmethod
    def convert_current(c):
        max_current = 100

        if 0 <= c <= max_current:
            scaled = c
        else:
            raise ValueError(f'Current outside range [0, {max_current}]')

        return 65535 * scaled / max_current

    def set_voltage(self, channel, v):
        ''' set voltage of channels 1-120, convert voltage values to bit values'''

        line_ind = XPOW.calc_line_ind(channel)

        if v > self.safe_voltage:
            raise ValueError(f'DO NOT EXCEED {self.safe_voltage}V')

        # The only safe mode of operation is first set voltage, and then sweep current. Voltage should keep fixed once
        # already set.
        if self.get_cache_voltage(channel) == 0 and self.get_cache_current(channel) == 0:
            self._cache_voltage[channel - 1] = v
        elif v - self.get_cache_voltage(channel) > self.voltage_step:
            raise ValueError(f'VOLTAGE STEP SHOULD NOT EXCEED {self.voltage_step}V')
        else:
            self._cache_voltage[channel - 1] = v

        voltage_b_val = self.convert_voltage(v)
        commandv = "CH:" + str(channel) + ":VOLT:" + str(int(voltage_b_val)) + "\n"

        # logging.info(commandv)

        # return self.SerialDatas[line_ind].write(commandv.encode())
        self.SerialDatas[line_ind].write(commandv.encode())
        sleep(0.005)

    def set_current(self, channel, c):

        line_ind = self.calc_line_ind(channel)

        if c > self.safe_current:
            raise ValueError(f'DO NOT EXCEED {self.safe_current}mA')

        if c - self.get_cache_current(channel) > self.current_step:
            raise ValueError(f'CURRENT STEP SHOULD NOT EXCEED {self.current_step}mA')
        else:
            self._cache_current[channel - 1] = c

        current_b_val = self.convert_current(c)
        commandc = "CH:" + str(channel) + ":CUR:" + str(int(current_b_val)) + "\n"

        # logging.info(commandc)

        # return self.SerialDatas[line_ind].write(commandc.encode())
        self.SerialDatas[line_ind].write(commandc.encode())
        sleep(0.005)

    def set_channel(self, channel, v, c):
        self.set_voltage(channel, v)
        self.set_current(channel, c)

    def read_channel(self, channel):
        '''Parse voltage and current data of some channel (1-140)'''
        line_ind = self.calc_line_ind(channel)

        command = "CH:" + str(channel) + ":VAL?\n"

        SerialData = self.SerialDatas[line_ind]
        SerialData.write(command.encode())
        sleep(0.1)

        while True:
            sleep(0.005)
            if (SerialData.inWaiting() > 0):
                myData = SerialData.readline().decode('utf-8')
                voltagevalue = myData.split(" ")[4]
                voltageparse = float(voltagevalue.replace('V,', ''))
                currentvalue = myData.split(" ")[5]
                currentparse = float(currentvalue.replace('mA\r\n', ''))
            else:
                break

        # print("Ch", channel, ":", voltageparse, "V")
        # print("Ch", channel, ":", currentparse, "mA\n")
        return (voltageparse, currentparse)

    def read_voltage(self, channel):
        '''Parse voltage data of some channel (1-140)'''
        voltageparse, _ = self.read_channel(channel)
        return voltageparse

    def read_current(self, channel):
        '''Parse voltage data of some channel (1-140)'''
        _, currentparse = self.read_channel(channel)
        return currentparse

    def zero_all(self):
        print('Zeroing all channels')
        for channel in range(1, 121):
            line_ind = self.calc_line_ind(channel)

            self.SerialDatas[line_ind].write(f"CH:{channel}:VOLT:0\n".encode())
            sleep(0.005)
            self.SerialDatas[line_ind].write(f"CH:{channel}:CUR:0\n".encode())
            sleep(0.005)

        print('All channels zeroed')

    def close(self):
        for SerialData in self.SerialDatas:
            SerialData.close()


class XPOWu(PowerSupply):

    def __init__(self, resource):
        '''
        :param resource: e.g. ['COM8', 'COM9', 'COM10']
        '''

        # write these into properties and make them immutable
        # ABSOLUTELY DO NOT CHANGE THESE PARAMETERS
        self._safe_voltage = 20
        self._safe_current = 100

        # DO NOT CHANGE THESE PARAMETERS
        self.line_num = 3
        self.channel_num = 40

        if len(resource) != self.line_num:
            raise ValueError('There should be three lines in total')

        # Connect device
        SerialDatas = []
        for com_channel in resource:
            # Search device through VID:PID for serial communication and stop the setting with last setting is bank
            # values 3
            SerialDatas.append(serial.Serial(com_channel, baudrate=115200, timeout=3.0, writeTimeout=0))

        # Sort according to board number
        board_order = []
        for i in range(3):
            SerialDatas[i].write(b'board?\n')
            while True:
                if (SerialDatas[i].inWaiting() > 0):
                    data = SerialDatas[i].readline().decode('utf-8')
                    board_number = int(data[1:-1].split(":")[1])
                    board_order.append(board_number)
                    break
        SerialDatas = [SerialDatas[i-1] for i in board_order]

        self.SerialDatas = SerialDatas
        self.wait_time = 0.005

        #TODO: add svr function and set default range to be 20
        self._range = {0: 5, 1: 10, 2: 20, 3: 32}

    @property
    def safe_voltage(self):
        return self._safe_voltage

    @property
    def safe_current(self):
        return self._safe_current

    @staticmethod
    def calc_line_ind(channel):
        ''' Calculate which line (0,1,2) and the corresponding channel index in that line does the channel (1-120)
        belong'''

        if 1 <= channel <= 40:
            line_ind = 0
        elif 41 <= channel <= 80:
            line_ind = 1
        elif 81 <= channel <= 120:
            line_ind = 2
        else:
            raise ValueError(f'channel_num not in range [1, 120]')

        channel = channel - line_ind * 40

        return line_ind, channel

    def _serial_wait(self, line_ind):
        # runs danger of an infinite loop, hence make private.
        sleep(self.wait_time)
        # Remove cached readout.
        while True:
            if (self.SerialDatas[line_ind].inWaiting() > 0):
                myData = self.SerialDatas[line_ind].readline()
                break

    def set_voltage(self, channel, v):

        line_ind, channel = self.calc_line_ind(channel)

        if v > self.safe_voltage:
            # raise ValueError(f'DO NOT EXCEED {self.safe_voltage}V')
            v = self.safe_voltage

        commandv = "CH:" + str(channel) + ":VOLT:" + str(v) + "\n"

        # return self.SerialDatas[line_ind].write(commandv.encode())
        self.SerialDatas[line_ind].write(commandv.encode())
        self._serial_wait(line_ind)

    def set_current(self, channel, c):

        line_ind, channel = self.calc_line_ind(channel)

        if c > self.safe_current:
            # raise ValueError(f'DO NOT EXCEED {self.safe_current}mA')
            c = self.safe_current

        commandc = "CH:" + str(channel) + ":CUR:" + str(c) + "\n"

        self.SerialDatas[line_ind].write(commandc.encode())
        self._serial_wait(line_ind)

    def set_channel(self, channel, v, c):
        self.set_voltage(channel, v)
        self.set_current(channel, c)

    def read_channel(self, channel):
        """
        Read voltage and current from channel
        :param channel: The channel to read from.
        :return: tuple (v,c)
        """

        line_ind, channel = self.calc_line_ind(channel)

        command = "CH:" + str(channel) + ":VAL?\n"
        self.SerialDatas[line_ind].write(command.encode())
        sleep(0.005)
        while True:
            if (self.SerialDatas[line_ind].inWaiting() > 0):
                myData = self.SerialDatas[line_ind].readline().decode('utf-8')
                voltageparse = float(myData[1:-1].split(":")[2])
                currentvalue = myData[1:-1].split(":")[3]
                currentparse = float(currentvalue.replace('>\r', ''))

                break

        return (voltageparse, currentparse)

    def read_current(self, channel):
        _, currentparse = self.read_channel(channel)
        return currentparse

    def read_voltage(self, channel):
        voltageparse, _ = self.read_channel(channel)
        return voltageparse

    def zero_all(self):
        print('Zeroing all channels')
        for line_ind in range(self.line_num):
            for channel in range(1, self.channel_num+1):
                self.SerialDatas[line_ind].write(f"CH:{channel}:CUR:0\n".encode())
                self._serial_wait(line_ind)
                self.SerialDatas[line_ind].write(f"CH:{channel}:VOLT:0\n".encode())
                self._serial_wait(line_ind)

        print('All channels zeroed')

    def cv_calibrate(self, channel, v_cal=3, c_cal=10):
        """
        CALIB command is used to recalibrate voltage control, especially for low-resistance loads. The calibration
        method is by injecting a small voltage and a small current into the device under test.
        The XPOW will inject a voltage with a maximum of v_cal V and a current with a maximum of c_cal mA.
        Please be aware that after the XPOW is turned off, the calibration setting will be reset.
        """

        line_ind, channel = self.calc_line_ind(channel)

        if v_cal > self.safe_voltage:
            v_cal = self.safe_voltage
        if c_cal > self.safe_current:
            c_cal = self.safe_current

        command = "CH:" + str(channel) + ":CALIB:" + str(v_cal) + ":" + str(c_cal) + "\n"
        self.SerialDatas[line_ind].write(command.encode())
        self._serial_wait(line_ind)

    def cv_calibrate_all(self, v_cal=3, c_cal=10):
        v_cal = np.atleast_1d(v_cal)
        if len(v_cal) == 1:
            v_cal = np.ones(self.line_num * self.channel_num) * v_cal
        elif len(v_cal) != self.line_num * self.channel_num:
            raise ValueError('Number of voltage values does not match 120')

        c_cal = np.atleast_1d(c_cal)
        if len(c_cal) == 1:
            c_cal = np.ones(self.line_num * self.channel_num) * c_cal
        elif len(c_cal) != self.line_num * self.channel_num:
            raise ValueError('Number of current values does not match 120')

        for i in range(self.line_num * self.channel_num):
            channel = i + 1
            self.cv_calibrate(channel, v_cal[i], c_cal[i])

        print('Calibration finished')

    def measure_config(self, volt_conv_t, curr_conv_t, averaging):
        """
        :param volt_conv_t: voltage conversion time per channel (us). Default: 588
        :param curr_conv_t: current conversion time per channel (us). Default: 588
        :param averaging: averaging sample count (num of samples). Default: 64
        """
        command = "MEASCONF" + ":" + str(volt_conv_t) + ":" + str(curr_conv_t) + ":" + str(averaging) + "\n"

        for line_ind in range(self.line_num):
            self.SerialDatas[line_ind].write(command.encode())
            self._serial_wait(line_ind)

        print('Measurement reconfigured. ')

    def close(self):
        for SerialData in self.SerialDatas:
            SerialData.close()