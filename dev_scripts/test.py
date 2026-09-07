# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import time

for i in range(20):
    setChannel(1, 20, i)
    time.sleep(.1)    
    readCommand1(1)
    
#%%
import sys

sys.path.append('C:\Program Files (x86)\IVI Foundation\VISA\WinNT\TLPM\Examples\Python')

from datetime import datetime
from ctypes import cdll,c_long, c_ulong, c_uint32,byref,create_string_buffer,c_bool,c_char_p,c_int,c_int16,c_double, sizeof, c_voidp
from TLPM import TLPM
import time


tlPM = TLPM()
deviceCount = c_uint32()
tlPM.findRsrc(byref(deviceCount))

print("devices found: " + str(deviceCount.value))

resourceName = create_string_buffer(1024)

for i in range(0, deviceCount.value):
    tlPM.getRsrcName(c_int(i), resourceName)
    print(c_char_p(resourceName.raw).value)
    break

tlPM.close()
#%%
tlPM = TLPM()
# resourceName = create_string_buffer(b"COM1::115200")
resourceName = create_string_buffer(b"USB0::0x1313::0x8078::P0020827::INSTR")
print(c_char_p(resourceName.raw).value)
tlPM.open(resourceName, c_bool(True), c_bool(False))
 
time.sleep(1) 

tlPM.writeRaw(c_char_p("*IDN?".encode('utf-8')))
response = create_string_buffer(1024)
retCount = c_uint32()
tlPM.readRaw(response, 1024, byref(retCount))
print(c_char_p(response.raw).value)

tlPM.close()
print('End program')

#%%


tlPM = TLPM()
resourceName = create_string_buffer(b"USB0::0x1313::0x8078::P0020827::INSTR")
print(c_char_p(resourceName.raw).value)
tlPM.open(resourceName, c_bool(True), c_bool(True))

message = create_string_buffer(1024)
tlPM.getCalibrationMsg(message)
print(c_char_p(message.raw).value)

time.sleep(5)
#%%
power_measurements = []
times = []
count = 0
while count < 20:
    power =  c_double()
    tlPM.measPower(byref(power))
    power_measurements.append(power.value)
    times.append(datetime.now())
    print(power.value)
    count+=1
    time.sleep(1)
#%%
tlPM.close()
print('End program')


#%%

import pyvisa
from ThorlabsPM100 import ThorlabsPM100
import time
rm = pyvisa.ResourceManager()
inst = rm.open_resource('USB0::0x1313::0x8078::P0020827::INSTR', timeout=1)
# power_meter = ThorlabsPM100(inst=inst)

#%%


inst.read_termination = '\n'
inst.write_termination = '\n'
inst.timeout = 1000
# pm = ThorlabsPM100(inst=inst)

#%%

inst.write('CORR:WAV 1550')  # set wavelength

for i in range(5):
    print(inst.query('MEAS:POW?'))
    time.sleep(1)


#%%
import serial  # Import the serial library
import csv
import time

##
##=========================================================
##

from serial.tools import list_ports
from time import sleep

##
##=========================================================
#
#   Example of controlling Xpow with Python through serial
#   Copyright Nicslab Pty Ltd
#   Updated 20210623
#   XPOW-120AX-CCvCV
#   *IDN?   - Retrieve device info
#   Example on creating simple command and set bank channels
#   Please email: support@nicslab.com for any enquiry
#
##

SerialData1 = serial.Serial('COM9', baudrate=115200, timeout=3.0,
                            writeTimeout=0)  # Search device through VID:PID for serial communication and stop the setting with last setting is bank values 3
SerialData2 = serial.Serial('COM10', baudrate=115200, timeout=3.0,
                            writeTimeout=0)  # Search device through VID:PID for serial communication and stop the setting with last setting is bank values 3
SerialData3 = serial.Serial('COM12', baudrate=115200, timeout=3.0,
                            writeTimeout=0)  # Search device through VID:PID for serial communication and stop the setting with last setting is bank values 3

# edit channel maximum of xpow per line
channelMax = 40

# Maximum Voltage per Channel
MaxVoltage = [29] * channelMax * 3

#Range Dictionary
Range = {0: 5, 1: 10, 2: 20, 3: 40}






