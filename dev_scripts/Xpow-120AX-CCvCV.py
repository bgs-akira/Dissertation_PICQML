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
#%%

# serVoltage1Channels: set voltage of channels 1 to 40, convert voltage values to bit values
def setVoltage1Channels(AllVValues):

    for channel, value in enumerate(AllVValues):
        
        if value > MaxVoltage[channel]:
            value = MaxVoltage[channel]

        value = (value * 65535 / MaxVoltage[channel])
        channel = channel + 1
        commandv = "CH:" + str(channel) + ":VOLT:" + str(int(value)) + "\n"
        print(commandv)

        SerialData1.write(commandv.encode())
        sleep(0.005)

    return


# serVoltage2Channels: set voltage of channels 41 to 80, convert voltage values to bit values
def setVoltage2Channels(AllVValues):
    
    for channel, value in enumerate(AllVValues):
        
        if value > MaxVoltage[channel+40]:
            value = MaxVoltage[channel+40]

        value = (value * 65535 / MaxVoltage[channel+40])
        channel = channel + 1
        commandv = "CH:" + str(channel) + ":VOLT:" + str(int(value)) + "\n"
        print(commandv)

        SerialData2.write(commandv.encode())
        sleep(0.005)

    return


# serVoltage3Channels: set voltage of channels 81 to 120, convert voltage values to bit values
def setVoltage3Channels(AllVValues):
    
    for channel, value in enumerate(AllVValues):
        
        if value > MaxVoltage[channel+80]:
            value = MaxVoltage[channel+80]

        value = (value * 65535 / MaxVoltage[channel+80])
        channel = channel + 1
        commandv = "CH:" + str(channel) + ":VOLT:" + str(int(value)) + "\n"
        print(commandv)

        SerialData3.write(commandv.encode())
        sleep(0.005)

    return


# setCurrent1Channels: set current of Channel 1 to 40, convert current values to bit values
def setCurrent1Channels(AllCValues):
    MaxCurrent = 300

    for y in AllCValues:
        if y > 300:
            y = 300
        else:
            y

    CurrentValues = ((y * 65535 / MaxCurrent) for y in AllCValues)

    for channel, value in enumerate(CurrentValues):
        channel = channel + 1
        commandc = "CH:" + str(channel) + ":CUR:" + str(int(value)) + "\n"
        print(commandc)

        SerialData1.write(commandc.encode())
        sleep(0.005)

    return


# setCurrent2Channels: set current of Channel 41 to 80, convert current values to bit values
def setCurrent2Channels(AllCValues):
    MaxCurrent = 300

    for y in AllCValues:
        if y > 300:
            y = 300
        else:
            y

    CurrentValues = ((y * 65535 / MaxCurrent) for y in AllCValues)

    for channel, value in enumerate(CurrentValues):
        channel = channel + 1
        commandc = "CH:" + str(channel) + ":CUR:" + str(int(value)) + "\n"
        print(commandc)

        SerialData2.write(commandc.encode())
        sleep(0.005)

    return


# setCurrent3Channels: set current of Channel 81 to 120, convert current values to bit values
def setCurrent3Channels(AllCValues):
    MaxCurrent = 300

    for y in AllCValues:
        if y > 300:
            y = 300
        else:
            y

    CurrentValues = ((y * 65535 / MaxCurrent) for y in AllCValues)

    for channel, value in enumerate(CurrentValues):
        channel = channel + 1
        commandc = "CH:" + str(channel) + ":CUR:" + str(int(value)) + "\n"
        print(commandc)

        SerialData3.write(commandc.encode())
        sleep(0.005)

    return

# setRange1Channels: Set Range for channel 1 - 40
def setRange1Channels(AllRangeValues):
   
    for y in AllRangeValues:
        if y > 3 and y < 0:
            y = 3
    
    for channel, value in enumerate(AllRangeValues):
        MaxVoltage[channel] = Range[value]
        channel = channel + 1
        commandr = "CH:" + str(channel) + ":SVR:" + str(int(value)) + "\n"
        print(commandr)

        SerialData1.write(commandr.encode())
        sleep(0.005)

    return

# setRange2Channels: Set Range for channel 41 - 80
def setRange2Channels(AllRangeValues):
   
    for y in AllRangeValues:
        if y > 3 and y < 0:
            y = 3
    
    for channel, value in enumerate(AllRangeValues):
        MaxVoltage[channel+40] = Range[value]
        channel = channel + 1
        commandr = "CH:" + str(channel) + ":SVR:" + str(int(value)) + "\n"
        print(commandr)

        SerialData2.write(commandr.encode())
        sleep(0.005)

    return

# setRange3Channels: Set Range for channel 81 - 120
def setRange3Channels(AllRangeValues):
   
    for y in AllRangeValues:
        if y > 3 and y < 0:
            y = 3
    
    for channel, value in enumerate(AllRangeValues):
        MaxVoltage[channel+80] = Range[value]
        channel = channel + 1
        commandr = "CH:" + str(channel) + ":SVR:" + str(int(value)) + "\n"
        print(commandr)

        SerialData3.write(commandr.encode())
        sleep(0.005)

    return



# readCommand1: read all values in Channel 1 to 40
def readCommand1(maxChannel):

    for channel in range(maxChannel):
        channel = channel + 1
        command = "CH:" + str(channel) + ":VAL?\n"

        SerialData1.write(command.encode())
        sleep(0.005)
        while True:
            if (SerialData1.inWaiting() > 0):
                myData = SerialData1.readline()
                print (myData)
            break

    return


# readCommand2: read all values in Channel 41 to 80
def readCommand2(maxChannel):

    for channel in range(maxChannel):
        channel = channel + 1
        command = "CH:" + str(channel) + ":VAL?\n"

        SerialData2.write(command.encode())
        sleep(0.005)
        while True:
            if (SerialData2.inWaiting() > 0):
                myData = SerialData2.readline()
                print (myData)
            break

    return


# readCommand3: read all values in Channel 81 to 120
def readCommand3(maxChannel):

    for channel in range(maxChannel):
        channel = channel + 1
        command = "CH:" + str(channel) + ":VAL?\n"

        SerialData3.write(command.encode())
        sleep(0.005)
        while True:
            if (SerialData3.inWaiting() > 0):
                myData = SerialData3.readline()
                print (myData)
            break

    return


# setChannel: set voltage and current of each channel
# setChannel(ch, voltage (in V), current (in mA))
def setChannel(channel, voltageVal, currentVal):
    MaxCurrent = 300
    
    if voltageVal > MaxVoltage[channel-1]:
        voltageVal = MaxVoltage[channel-1]  

    CurrentValues = currentVal * 65535 / MaxCurrent
    VoltageValues = voltageVal * 65535 / MaxVoltage[channel-1]

    if ((channel >= 1) and (channel <= 40)):
        commandv = "CH:" + str(channel) + ":VOLT:" + str(int(VoltageValues)) + "\n"
        commandc = "CH:" + str(channel) + ":CUR:" + str(int(CurrentValues)) + "\n"

        SerialData1.write(commandc.encode())
        sleep(0.005)
        SerialData1.write(commandv.encode())
    elif ((channel >= 41) & (channel <= 80)):
        channel = channel - 40
        commandv = "CH:" + str(channel) + ":VOLT:" + str(int(VoltageValues)) + "\n"
        commandc = "CH:" + str(channel) + ":CUR:" + str(int(CurrentValues)) + "\n"

        SerialData2.write(commandc.encode())
        sleep(0.005)
        SerialData2.write(commandv.encode())
    else:
        channel = channel - 80
        commandv = "CH:" + str(channel) + ":VOLT:" + str(int(VoltageValues)) + "\n"
        commandc = "CH:" + str(channel) + ":CUR:" + str(int(CurrentValues)) + "\n"

        SerialData3.write(commandc.encode())
        sleep(0.005)
        SerialData3.write(commandv.encode())

    return


# setZeroAllChannels: set all channels to zero
def setZeroAllChannels():
    for channel in range(channelMax):
        channel = channel + 1
        commandc = "CH:" + str(channel) + ":CUR:0\n"
        commandv = "CH:" + str(channel) + ":VOLT:0\n"

        SerialData1.write(commandc.encode())
        sleep(0.005)
        SerialData1.write(commandv.encode())

    for channel in range(channelMax):
        channel = channel + 1
        commandc = "CH:" + str(channel) + ":CUR:0\n"
        commandv = "CH:" + str(channel) + ":VOLT:0\n"

        SerialData2.write(commandc.encode())
        sleep(0.005)
        SerialData2.write(commandv.encode())

    for channel in range(channelMax):
        channel = channel + 1
        commandc = "CH:" + str(channel) + ":CUR:0\n"
        commandv = "CH:" + str(channel) + ":VOLT:0\n"

        SerialData3.write(commandc.encode())
        sleep(0.005)
        SerialData3.write(commandv.encode())

    return

# Parse Voltage Data Example
def parseVoltage(channel):
    if channel <= 40:
        SerialData = SerialData1
    elif channel <= 80:
        SerialData = SerialData2
    else:
        SerialData = SerialData3

    channel = (channel-1) % 40 + 1
    command = "CH:" + str(channel) + ":VAL?\n"

    SerialData.write(command.encode())
    sleep(0.005)
    while True:
        if (SerialData.inWaiting() > 0):
            myData = SerialData.readline().decode('utf-8')
            voltagevalue = myData.split(" ")[4]
            voltagefloat = float(voltagevalue.replace('V,', ''))
            print(voltagefloat)
        break

    return

# Parse Current Data Example
def parseCurrent(channel):
    if channel <= 40:
        SerialData = SerialData1
    elif channel <= 80:
        SerialData = SerialData2
    else:
        SerialData = SerialData3

    channel = (channel-1) % 40 + 1
    command = "CH:" + str(channel) + ":VAL?\n"

    SerialData.write(command.encode())
    sleep(0.005)
    while True:
        if (SerialData.inWaiting() > 0):
            myData = SerialData.readline().decode('utf-8')
            currentvalue = myData.split(" ")[5]
            currentfloat = float(currentvalue.replace('mA\r\n', ''))
            print(currentfloat)
        break

    return

# Parse All Data Example
def parseValue(channel):
    if channel <= 40:
        SerialData = SerialData1
    elif channel <= 80:
        SerialData = SerialData2
    else:
        SerialData = SerialData3

    channel = (channel-1) % 40 + 1
    command = "CH:" + str(channel) + ":VAL?\n"

    SerialData.write(command.encode())
    sleep(0.005)
    while True:
        if (SerialData.inWaiting() > 0):
            myData = SerialData.readline().decode('utf-8')
            voltagevalue = myData.split(" ")[4]
            voltageparse = float(voltagevalue.replace('V,', ''))
            currentvalue = myData.split(" ")[5]
            currentparse = float(currentvalue.replace('mA\r\n', ''))

        break

    print("Ch", channel,":", voltageparse, "V")
    print("Ch", channel,":", currentparse, "mA\n")

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
        fieldnames = ['voltage(v)', 'current(mA)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for value in range(len(readValueV)):
            writer.writerow({'voltage(v)': readValueV[value], 'current(mA)': readValueC[value]})

    print("function end")

    return

#%%
# Edit input voltage with value from 0 - 32 V
# Example: InputV = [1.2, 2.3, 3.4, 4.5, 5.7, 6.8, 7.2, 8]
# Edit InputV to set the voltage of each channel
# InputV1 = Channel 1 to 40
InputV1 = [30, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
           28, 27, 26, 25, 24, 23, 22, 21, 20, 19]
# InputV2 = Channel 41 to 80
InputV2 = [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
           14, 15, 16, 17, 18, 19, 20, 21]
# InputV3 = Channel 81 to 120
InputV3 = [22, 23, 24, 25, 26, 27, 28, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
           9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3]

# Edit input current with value from 0 - 300 mA
# Example:InputC = [300, 300, 300, 300, 300, 300, 300, 300]
# Edit InputC to set the current of each channel
# InputC1 = Channel 1 to 40
InputC1 = [300, 270, 243, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300,
           300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
# InputC2 = Channel 41 to 80
InputC2 = [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300,
           300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]
# InputC3 = Channel 81 to 120
InputC3 = [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300,
           300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300]

# Edit input range with value from 0 - 3
# Range mode
# 0. 0 - 5V
# 1. 0 - 10V
# 2. 0 - 20V
# 3. 0 - 40V 
# Example:InputR = [0, 0, 0, 0, 1, 1, 1, 1]
# Edit InputR to set the Voltage Range of each channel
# InputR1 = Channel 1 to 40
InputR1 = [0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
# InputR1 = Channel 1 to 40
InputR2 = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1]
# InputR1 = Channel 1 to 40
InputR3 = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
# Example Code 1 ---------------------------------------------------------------------

# setVoltage1Channels(InputV1)
# setCurrent1Channels(InputC1)
# setVoltage2Channels(InputV2)
# setCurrent2Channels(InputC2)
# setVoltage3Channels(InputV3)
# setCurrent3Channels(InputC3)
# sleep(0.5)
# readCommand1(channelMax)
# sleep(0.5)
# readCommand2(channelMax)
# sleep(0.5)
# readCommand3(channelMax)
# sleep(1)

# Example Code 1 - end  ---------------------------------------------------------------------

# Example Code 2 ---------------------------------------------------------------------

# set channel 1 with 5 V and 300 mA
# setChannel(2, 30, 300)
# sleep(0.5)
# parseValue(2)

# Example Code 2 - end  ---------------------------------------------------------------------

# Example Code 3 -----------------------------------------------------------------------

# channel = 1
# seqValueV = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# seqValueC = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
# duration = 1
# sweepOne(channel, seqValueV, seqValueC, duration)
#
# sleep(1)
# channel = 80
# seqValueV = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4]
# seqValueC = [100] * 10
# duration = 1
# sweepOne(channel, seqValueV, seqValueC, duration)

# Example Code 3 - end  ----------------------------------------------------------------

# Example Code 4 -----------------------------------------------------------------------

setRange2Channels(InputR2)
setVoltage2Channels(InputV2)
setCurrent2Channels(InputC2)
sleep(0.5)
readCommand2(channelMax)


# Example Code 4 - end  ----------------------------------------------------------------
