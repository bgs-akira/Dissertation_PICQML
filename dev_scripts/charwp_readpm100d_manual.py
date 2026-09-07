#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Get power reading from Thorlabs PM100D"""
"""To characterize waveplates"""

from ThorlabsPM100 import ThorlabsPM100, USBTMC #install with py -3 -m pip install ThorlabsPM100 or sth like this
import argparse
import sys
import csv
import numpy, scipy, scipy.optimize
from scipy.signal import argrelmax, argrelmin
import matplotlib
import matplotlib.pyplot as plt
import array
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle
import os

parser = argparse.ArgumentParser(description='Characterize waveplates with a PM100D powermeter.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-o', '--waveplate-name' , type=str, help='String to identify waveplate', default="")
parser.add_argument('-p', '--powermeter-device' , type=str, help='path to usbtmc device', default='/dev/usbtmc0')
parser.add_argument('-n', '--serial-number' , type=str, help='Serial number of powermeter (use on windows)', default='P0007675')
parser.add_argument('-s', '--save-plot' , type=bool, help='Serial number of powermeter (use on windows)', default=True)
args = parser.parse_args()
saveplot=args.save_plot

if sys.platform != 'linux':
    WINDOWS=True
    import pyvisa as visa
    eol='\r\n'
    if args.serial_number == "":
        print("Please provide a powermeter serial number")
        sys.exit()
    else:
        sn = args.serial_number
else:
    WINDOWS=False
    if args.powermeter_device == "":
        print("Please provide a powermeter device path")
        sys.exit()
    else:
        pmdevice = args.powermeter_device

# get file/waveplate name
if args.waveplate_name == "" :
    askstr="Please enter a waveplate name: "
    wpname=input(askstr)
    if len(wpname)==0:
        wpname="wp"
    wpname=wpname.replace(" ","_")
else:
    wpname=args.waveplate_name.replace(" ","_")

fname=wpname+".csv"
pi=numpy.pi

# fit functions
def sinfunc(param,x):
    return param[0]+param[1]*numpy.sin(param[2]*x+param[3])
def sinfuncerr(param,x,y):
    return sinfunc(param,x)-y

#open file
f=open(fname, mode='w', newline='\n')
fwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
fwriter.writerow(["ang",wpname])

angles=numpy.arange(0,361,10)
power=numpy.zeros(0)
header=[]
indat=[]

#connect to powermeter powermeter
if WINDOWS:
    rm = visa.ResourceManager()
    inst=rm.open_resource("USB0::0x1313::0x8078::"+sn+"::INSTR")
    inst.read_termination = '\n'
    inst.write_termination = '\n'
    inst.timeout = 1000
    pm = ThorlabsPM100(inst=inst)
else:
    inst=USBTMC(device=pmdevice)
    pm=ThorlabsPM100(inst=inst)
pm.sense.average.count=20
PuW = -1
PW = pm.read
PuW = PW*1000000

#get data...
for ang in angles:
	input("set waveplate to "+ str(ang) +" deg.\nPress any key.")
	PW=pm.read*1000000
	print(PW)
	power=numpy.append(power,PW)
	fwriter.writerow([ang,PW])
f.close()
if WINDOWS:
    rm.close()

with open(fname, 'r', newline='\n') as infile:
    tmpreader = csv.reader(infile, delimiter=',')
    for row in tmpreader:
        indat.append(row)

header=indat[0][1:]
angles=numpy.transpose(indat)[0][1:]
angles2=indat[1:][0]

wp=[]

for i in range(1,len(indat)):
    tmp=[]
    for j in range(1,len(indat[i])):
        tmp.append(float(indat[i][j]))
    wp.append(tmp)

header=indat[0][1:]

angles=[]
for i in range(1,len(indat)):
    angles.append(float(indat[i][0]))

wp=numpy.array(wp).T.tolist()
angles=numpy.asarray(angles)

plotrange=numpy.arange(0,360,0.001)

for i in range(0,len(wp)):
	#fit
    parest=[1.,max(wp[i]),1/(8*pi),0.]
    par,success=scipy.optimize.leastsq(sinfuncerr, parest[:], args=(angles,wp[i]))

	#plot
    fig, ax1 = plt.subplots()
    ax1.plot(angles,wp[i],linestyle='none',marker='o',ms=3)
    ax1.plot(plotrange,sinfunc(par,plotrange))
    plt.title('Waveplate ' + header[i], fontsize=14, fontweight='bold')
    ax1.set_xlabel('Rotation angle [°]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Power, H polarized (a.u.)', fontsize=12, fontweight='bold')
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    maximaindices=argrelmax(sinfunc(par,plotrange), order=20)
    maxtext='Maxima at '
    maxima=[]
    for j in maximaindices[0]:
        maxima.append(plotrange[j])
    for k in range(0,len(maxima)):
        maxtext=maxtext+'{0:.2f}°, '.format(maxima[k])
    maxtext=maxtext[:-2]
    print(maxima)
    fig.tight_layout()
    ax1.text(0.01,0.01,maxtext, fontsize='8',color='black',transform=ax1.transAxes)
	#save to file
    if saveplot:
        plotsavename=wpname
        print('saving ' + plotsavename)
        plt.savefig(plotsavename+'.pdf', format="pdf",transparent=True, bbox_inches='tight', pad_inches=0)
        plt.savefig(plotsavename+'.png', format="png",transparent=True, bbox_inches='tight', pad_inches=0)
        plt.show()
