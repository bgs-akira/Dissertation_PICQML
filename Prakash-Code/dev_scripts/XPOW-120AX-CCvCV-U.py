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

serialData1 = serial.Serial('COM12', baudrate=115200, timeout=3.0,
                            writeTimeout=0)  # Search device through VID:PID for serial communication and stop the setting with last setting is bank values 3
serialData2 = serial.Serial('COM13', baudrate=115200, timeout=3.0,
                            writeTimeout=0)  # Search device through VID:PID for serial communication and stop the setting with last setting is bank values 3
serialData3 = serial.Serial('COM14', baudrate=115200, timeout=3.0,
                            writeTimeout=0)  # Search device through VID:PID for serial communication and stop the setting with last setting is bank values 3

serialData = [serialData1, serialData2, serialData3]

# edit channel maximum of Xpow
channelMax = 120

# Maximum Voltage per Channel
MaxVoltage = [40] * channelMax

# Range Dictionary
Range = {0: 5, 1: 10, 2: 20, 3: 32}


def checkBoard():
    command = "board?\n"

    portBuffer = [serialData1, serialData2, serialData3]

    for i in range(3):
        portBuffer[i].write(command.encode())
        while True:
            if (portBuffer[i].inWaiting() > 0):
                myData = portBuffer[i].readline().decode('utf-8')
                boardNumber = myData[1:-1].split(":")[1]
                boardNumber = int(boardNumber.replace('>\r', ''))
                print(boardNumber)
                break

        serialData[boardNumber - 1] = portBuffer[i]

    # setVoltageAllChannels: set voltage of all channels, convert voltage values to bit values


def setVoltageAllChannels(AllVValues):
    for i in range(3):
        commandv_list = ["CH:1-5", "CH:6-10", "CH:11-15", "CH:16-20", "CH:21-25", "CH:26-30", "CH:31-35", "CH:36-40"]

        for tab in range(8):
            commandv = commandv_list[tab] + ":VOLT"
            lindex = (tab * 5) + i * 40
            hindex = ((tab + 1) * 5) + i * 40

            for channel, value in enumerate(AllVValues[lindex:hindex]):

                if value > MaxVoltage[channel + i * 40]:
                    value = MaxVoltage[channel + i * 40]

                commandv += (":" + str(value))

            commandv += "\n"

            serialData[i].write(commandv.encode())
            sleep(0.05)

    return


# setCurrentAllChannels: set current of all channels, convert current values to bit values
def setCurrentAllChannels(AllCValues):
    MaxCurrent = 300

    for i in range(3):
        commandc_list = ["CH:1-5", "CH:6-10", "CH:11-15", "CH:16-20", "CH:21-25", "CH:26-30", "CH:31-35", "CH:36-40"]

        for tab in range(8):
            commandc = commandc_list[tab] + ":CUR"
            lindex = (tab * 5) + i * 40
            hindex = ((tab + 1) * 5) + i * 40

            for channel, value in enumerate(AllCValues[lindex:hindex]):

                if value > MaxCurrent:
                    value = MaxCurrent

                commandc += (":" + str(value))

            commandc += "\n"

            serialData[i].write(commandc.encode())
            sleep(0.05)

    return


# setRangeAllChannels: set range of all channels
def setRangeAllChannels(AllRangeValues):
    for y in AllRangeValues:
        if y > 3 and y < 0:
            y = 3

    for channel, value in enumerate(AllRangeValues):
        MaxVoltage[channel] = Range[value]
        channel = channel + 1

        idx = int((channel - 1) / 40)
        channel = channel - idx * 40

        commandr = "CH:" + str(channel) + ":SVR:" + str(int(value)) + "\n"
        print(commandr)

        serialData[idx].write(commandr.encode())
        serialWait(idx + 1)

    return


# setOffAllChannels: set all channels to zero
def setOffAllChannels(maxChannel):
    for i in range(3):
        for channel in range(maxChannel):
            channel = channel + 1
            commandc = "CH:" + str(channel) + ":CUR:0\n"
            commandv = "CH:" + str(channel) + ":VOLT:0\n"

            serialData[i].write(commandc.encode())
            serialWait(i + 1)
            serialData[i].write(commandv.encode())
            serialWait(i + 1)

    return


# readAllChannels: read all values in each channel
def readMultipleChannels():
    for i in range(3):
        print("Measurement for board: " + str(i + 1))
        for channel in range(40):
            channel = channel + 1
            command = "CH:" + str(channel) + ":VAL?\n"

            serialData[i].write(command.encode())
            sleep(0.005)
            while True:
                if (serialData[i].inWaiting() > 0):
                    myData = serialData[i].readline()
                    print(myData)
                    break

    return


# setChannel: set voltage and current of each channel
# setChannel(ch, voltage (in V), current (in mA))
def setChannel(channel, voltageVal, currentVal):
    MaxCurrent = 300
    if voltageVal > MaxVoltage[channel - 1]:
        voltageVal = MaxVoltage[channel - 1]

    if currentVal > MaxCurrent:
        currentVal = MaxCurrent

    CurrentValues = currentVal
    VoltageValues = voltageVal

    idx = int((channel - 1) / 40)
    channel = channel - idx * 40

    commandv = "CH:" + str(channel) + ":VOLT:" + str(VoltageValues) + "\n"
    commandc = "CH:" + str(channel) + ":CUR:" + str(CurrentValues) + "\n"

    serialData[idx].write(commandc.encode())
    serialWait(idx + 1)
    serialData[idx].write(commandv.encode())
    serialWait(idx + 1)

    return


# setOff:  set one channel to zero
# setOff(1) -> set 0 V, 0 mA, to channel 1.
def setOff(channel):
    idx = int((channel - 1) / 40)
    channel = channel - idx * 40

    commandc = "CH:" + str(channel) + ":CUR:0\n"
    commandv = "CH:" + str(channel) + ":VOLT:0\n"

    serialData[idx].write(commandc.encode())
    serialWait(idx + 1)
    serialData[idx].write(commandv.encode())
    serialWait(idx + 1)

    return


# readChannel: read all values in each channel
def readChannel(channel):
    idx = int((channel - 1) / 40)
    channel = channel - idx * 40

    command = "CH:" + str(channel) + ":VAL?\n"

    serialData[idx].write(command.encode())
    sleep(0.005)
    while True:
        if (serialData[idx].inWaiting() > 0):
            myData = serialData[idx].readline()
            print(myData)
            break

    return myData


# Parse Voltage Data Example
def parseVoltage(channel):
    idx = int((channel - 1) / 40)
    channel = channel - idx * 40

    command = "CH:" + str(channel) + ":VAL?\n"
    serialData[idx].write(command.encode())
    sleep(0.005)
    while True:
        if (serialData[idx].inWaiting() > 0):
            myData = serialData[idx].readline().decode('utf-8')
            voltagefloat = float(myData[1:-1].split(":")[2])
            print(voltagefloat)
        break

    return


# Parse Current Data Example
def parseCurrent(channel):
    idx = int((channel - 1) / 40)
    channel = channel - idx * 40

    command = "CH:" + str(channel) + ":VAL?\n"
    serialData[idx].write(command.encode())
    sleep(0.005)
    while True:
        if (serialData[idx].inWaiting() > 0):
            myData = serialData[idx].readline().decode('utf-8')
            currentvalue = myData[1:-1].split(":")[3]
            currentfloat = float(currentvalue.replace('>\r', ''))
            print(currentfloat)
        break

    return


# Parse All Data Example
def parseValue(channel):
    idx = int((channel - 1) / 40)
    channel = channel - idx * 40

    command = "CH:" + str(channel) + ":VAL?\n"
    serialData[idx].write(command.encode())
    sleep(0.005)
    while True:
        if (serialData[idx].inWaiting() > 0):
            myData = serialData[idx].readline().decode('utf-8')
            voltageparse = float(myData[1:-1].split(":")[2])
            currentvalue = myData[1:-1].split(":")[3]
            currentparse = float(currentvalue.replace('>\r', ''))

            break

    # print("Ch", channel,":", voltageparse, "V")
    # print("Ch", channel,":", currentparse, "mA")

    return (voltageparse, currentparse)


# set one channel to run automatically and record it.
# duration in seconds
# example:
# seqValueV = [1, 2, 3, 4, 5, 6, 7, 8]
# seqValueC= [10, 10, 10, 10, 10, 10, 10, 10]
# sweepOne (1, seqValueV, seqValueC, 5)
# set channel 1 to change voltage value based on seqValueV and current value based seqValueC, for every 5 seconds
def sweepOne(channel, seqValueV, seqValueC, duration):
    readValueV = []
    readValueC = []
    for valuev, valuec in zip(seqValueV, seqValueC):
        setChannel(channel, valuev, valuec)
        sleep(duration)
        parseVal = parseValue(channel)
        readValueV.append(parseVal[0])
        readValueC.append(parseVal[1])

    with open('data.csv', mode='w') as csv_file:
        fieldnames = ['setV(v)', 'voltage(v)', 'setC(mA)', 'current(mA)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for value in range(len(readValueV)):
            writer.writerow({'setV(v)': seqValueV[value], 'voltage(v)': readValueV[value], 'setC(mA)': seqValueC[value],
                             'current(mA)': readValueC[value]})

    print("function end")

    return


# Set GPIO Digital Output
# Index = 12, 13, 16, 19, 26
# value = "HIGH" or "LOW"
def setGPIO(index, value):
    command = "GPIO:" + str(index) + ":" + value + "\n"
    serialData[1].write(command.encode())
    sleep(0.005)

    return


def cvCalibrationSingleChannel(channel, vcal, ccal):
    vvalue = vcal
    cvalue = ccal

    idx = int((channel - 1) / 40)
    channel = channel - idx * 40

    command = "CH:" + str(channel) + ":CALIB:" + str(vvalue) + ":" + str(cvalue) + "\n"
    serialData[idx].write(command.encode())

    while True:
        if (serialData[idx].inWaiting() > 0):
            myData = serialData[idx].readline()
            print(myData)
            break

    return


def cvCalibrationAllChannel(vcal, ccal):
    for i in range(channelMax):
        channel = i + 1
        cvCalibrationSingleChannel(channel, vcal[i], ccal[i])

    print("Calibration Finished")


def measurementConfig(voltConvTime, currConvTime, averaging):
    command = "MEASCONF" + ":" + str(voltConvTime) + ":" + str(currConvTime) + ":" + str(averaging) + "\n"

    for i in range(3):
        serialData[i].write(command.encode())

        while True:
            if (serialData[i].inWaiting() > 0):
                myData = serialData[i].readline()
                print(myData)
                break


def serialWait(board):
    sleep(0.005)
    while True:
        if (serialData[board - 1].inWaiting() > 0):
            myData = serialData[board - 1].readline()
            break


# Edit input voltage with value from 0 - 32 V
# Example: InputV = [1, 2, 3, 4, 5, 6, 7, 8]
# Edit InputV to set the voltage of each channel
InputV = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 28,
          27, 26, 25, 24, 23, 22, 21, 20, 19, 18] * 3

# Edit input current with value from 0 - 300 mA
# Example:InputC = [5, 10, 50, 3, 150, 200, 250, 300]
# Edit InputC to set the current of each channel
InputC = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
          10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10] * 3

# Edit input range with value from 0 - 3
# Range mode
# 0. 0 - 5V
# 1. 0 - 10V
# 2. 0 - 20V
# 3. 0 - 40V
# Example:InputR = [0, 0, 0, 0, 1, 1, 1, 1]
# Edit InputR to set the Voltage Range of each channel
InputR = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1,
          1, 1, 1] * 3

# Calibration Example
# vcal = [0.5] * channelMax
# ccal = [1] * channelMax
# cvCalibrationAllChannel(vcal, ccal)

# Measurement configuration
# Change all conversion time to 144 uS and averaging sample to 1 sample
# measurementConfig(144, 144, 1)

# Change all conversion time to 512 uS and averaging sample to 16 sample
# measurementConfig(512, 512, 16)

# Arrange Serial Communication for Xpow-120AX-CCvCV-U
checkBoard()

# Example Code 1 -----------------------------------------------------------------------

# setVoltageAllChannels([5]*channelMax)
# setCurrentAllChannels([3]*channelMax)
# sleep(1)
# readMultipleChannels()

# Example Code 1 - end  ----------------------------------------------------------------


# Example Code 2 -----------------------------------------------------------------------
# set channel 1 with 5 V and 150 mA

# setChannel(120, 5, 1.5)
# sleep(0.5)
# parseVoltage(120)
# parseCurrent(120)

# setOff(120)
# sleep(0.5)
# parseVoltage(120)
# parseCurrent(120)

# Example Code 2 - end  ----------------------------------------------------------------

# Example Code 3 -----------------------------------------------------------------------

# channel = 5
# seqValueV = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# seqValueC = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
# duration = 1
# sweepOne(channel, seqValueV, seqValueC, duration)

# sleep(1)
# channel = 81
# seqValueV = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# seqValueC = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
# duration = 1
# sweepOne(channel, seqValueV, seqValueC, duration)
#
# setOff(channel)

# Example Code 3 - end  ----------------------------------------------------------------

# Example Code 4 -----------------------------------------------------------------------

# setRangeAllChannels([3] * channelMax)
# setChannel(1, 15, 3)
# sleep(0.5)
# print(parseValue(1))

# Example Code 4 - end  ----------------------------------------------------------------
