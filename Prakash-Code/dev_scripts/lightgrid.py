__author__ = "Nicholas C. Harris"
__copyright__ = "Copyright 2015"
__version__ = "1.0"
__maintainer__ = "Nicholas C. Harris"
__email__ = "harris.nicholasc@gmail.com"

import time
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import serial

# qt5 to real-time plotting
from matplotlib.backends.qt_compat import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure#

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        self.dynamic_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        layout.addWidget(self.dynamic_canvas)
        layout.addWidget(NavigationToolbar(self.dynamic_canvas, self))

class lightGridWrapper:

    def __init__(self, port):
        self.chans = 26
        self.port = port
        
        self.lg = serial.Serial(self.port, timeout = 0.1)
        self.lg.bautrate=115200
        
        self.lg.write(b'z') # reset

        self.gain_setting = 3
        self.set_gain(self.gain_setting)

        self.vp_lut = self.build_v_p_lut()

        self.darks = {}
        self.darks[3] = [5.6100000000000002e-05,
                         0.0001583,
                         4.6999999999999999e-06,
                         0.00061180000000000002,
                         0.00044270000000000003,
                         0.0011395000000000001,
                         4.4100000000000001e-05,
                         0.0015035000000000001,
                         0.00046509999999999998,
                         0.00059299999999999999,
                         7.2200000000000007e-05,
                         0.00061109999999999995,
                         0.00016750000000000001,
                         0.00090729999999999999,
                         7.47e-05,
                         0.0008206,
                         0.00084489999999999999,
                         0.0,
                         0.00027700000000000001,
                         0.0,
                         0.00091200000000000005,
                         2.7e-06,
                         4.9999999999999998e-07,
                         5.8699999999999997e-05,
                         0.00043399999999999998,
                         7.1299999999999998e-05]
        self.darks[3] = self.darks[3][:self.chans]
        self.darks[4] = [1.9e-06,
                         1.9400000000000001e-05,
                         2.2500000000000001e-05,
                         0.00028509999999999999,
                         0.00022589999999999999,
                         0.00043189999999999998,
                         1.3e-06,
                         0.00081079999999999998,
                         0.00048020000000000002,
                         0.00037359999999999997,
                         0.00070290000000000001,
                         0.00046050000000000003,
                         0.00079420000000000001,
                         0.00028160000000000001,
                         0.00064849999999999999,
                         0.00068559999999999997,
                         0.00044650000000000001,
                         0.0,
                         0.0002041,
                         0.0,
                         0.00048040000000000002,
                         1.7999999999999999e-06,
                         7.9999999999999996e-07,
                         3.3899999999999997e-05,
                         0.00022589999999999999,
                         4.1699999999999997e-05,
                         0.0002433,
                         0.0,
                         0.0,
                         0.00053149999999999996,
                         8.7100000000000003e-05,
                         0.0]
        self.darks[4] = self.darks[4][:self.chans]

        for gain_setting in self.darks:
            self.darks[gain_setting] = np.array(self.darks[gain_setting])

        # check the power level
        p0 = 1e6 * np.sum(self.read(10))
        print('---Starting total optical power (%f) uW---' % p0)
        if p0 < 0.1:
            print('!----Warning, input power to LightGrid is low (%f) uW----!' % p0)
        elif p0 > 15.:
            print('!----Warning, input power to LightGrid is high (%f) uW----!' % p0)

    def set_gain(self, g):
        '''
        Set the photodiode array gain.
          0: +/- 2.5 * VREF
          1: +/- 1.25 * VREF
          2: +/- 0.625 * VREF
          3: 0->2.5 * VREF
          4: 0->1.25 * VREF
        '''
        if g in [0, 1, 2, 3, 4]:
            self.gain_setting = g
            self.lg.write(b'g %d\n' % g)
        else:
            print('Invalid gain, must be in [1,2,3,4].')

    def read_with_var(self, N=1):
        p = [self.read(1) for n in range(N)]
        mean_p = np.mean(p, axis=0)*1e6
        std_p = np.std(p, axis=0)*1e6
        return mean_p, std_p

    def read(self, N=1, convert=True):
        '''
        Read all 12 photodiode channels. Return array
        will be a float proportional to actual power.

        N: number of averages constituting a reading.
        '''

        # read the first full array
        self.lg.write(b'p %d\n' % (N))

        # get data from full and half array
        data = list(map(float, (self.lg.readline())[0:-2].split(b',')))
        # data = list()
        data = data[:26]

        # subtract background
        # data -= self.darks[self.gain_setting]

        # convert to watts
        if convert:
            for i in range(len(data)):
                if i in self.vp_lut:
                    data[i] = self.vp_lut[i](data[i])
                else:
                    data[i] = 0.

        # ensure all values are positive
        for i, di in enumerate(data):
            data[i] = data[i].item()
            if di < 0:
                print(data[i], 'less than zero.')

        return data

    def __del__(self):
        self.lg.write(b'z')
        self.lg.close()

    def live_plot(self, sel=None):
        if sel == None:
            sel = range(self.chans)


        qapp = QtWidgets.QApplication.instance()
        if not qapp:
            qapp = QtWidgets.QApplication(sys.argv)

        app = ApplicationWindow()
        # app.addToolBar(QtCore.Qt.BottomToolBarArea,
                        # NavigationToolbar(app.dynamic_canvas, app))
        app._dynamic_ax = app.dynamic_canvas.figure.subplots()
        app._dynamic_ax.grid(True)

        app._timer = app.dynamic_canvas.new_timer(50)

            
        app._bar = app._dynamic_ax.bar(
            [str(i) for i in sel],
            height = np.zeros_like(sel),
            yerr = np.zeros_like(sel),
            ecolor='black',
            capsize=0,
            alpha=.5)
        
        app._text = app._dynamic_ax.annotate(' ', (.8,1.05), xycoords='axes fraction')

        app._dynamic_ax.set_xlabel('Detector')
        app._dynamic_ax.set_ylabel('Power (uW)')
            
        # handle the container of bar patches and errbar lines
        bar = app._bar.patches
        data_line, caplines, barlinecols = app._bar.errorbar.lines
        errbar = barlinecols[0]
                
        def _update_canvas(app):
            y, yerr = self.read_with_var(10)
            # filter the chanel of interst
            y = y[sel]
            yerr = yerr[sel]
            y_t = np.sum(y)
            for i in range(len(sel)):
                bar[i].set_height(y[i])
            errbar.set_segments(np.array(
                [ [ [i, y[i]+yerr[i]], [i, y[i]-yerr[i]] ] for i in range(len(sel)) ]
            ))
            app._dynamic_ax.set_ylim((0, (max(y+yerr)) ))
            app._text.set_text(f'total power {y_t:.3f} uW')
            app.dynamic_canvas.draw()

        app._timer.add_callback(_update_canvas, app)

        app._timer.start()
        app.show()
        app.activateWindow()
        app.raise_()
        qapp.exec_()

    def live_plot_old(self, sel=None):
        foo = True
        try:
            plt.ion()
            while foo ==True:
                plt.pause(0.05)
                plt.cla()
                y = self.read(10)

                if sel == None:
                    x = list(range(self.chans))
                else:
                    x = sel
                    y = y[sel]

                plt.bar(x, np.array(y) * 1e6, alpha=.5, edgecolor='none')
                plt.xlim([0, self.chans])
                plt.xlabel('Detector')
                plt.ylabel('Power (uW)')
                plt.draw()
        except KeyboardInterrupt:
            foo = False
            sys.exit()

    def time_trace(self, traces=[0]):
        '''
        Looks like a cardiograph.
        '''
        qapp = QtWidgets.QApplication.instance()
        if not qapp:
            qapp = QtWidgets.QApplication(sys.argv)

        app = ApplicationWindow()
        app._dynamic_axs = app.dynamic_canvas.figure.subplots(nrows=len(traces), ncols=1, sharex = True)
        # app.dynamic_canvas.figure.subplots_adjust(hspace=0)
        for ax in app._dynamic_axs:
            ax.grid(True)
            # ax.set_ylim(ymin=-50)
        app.t, app.y = np.arange(200), np.zeros((len(traces),200))

        app._lines = [ ax.plot(app.t, np.zeros_like(app.t),
            color='black',
            # capsize=0,
            alpha=.5)[0]
            for ax in app._dynamic_axs ]


        def _update_canvas(app):
            out = np.array(self.read(10))
            out = 10 * np.log10(out) + 30
            dy = out[traces]
            
            for i in range(len(traces)):
                line = app._lines[i]
                app.y[i] = np.append(app.y[i], dy[i])[1:]
                line.set_ydata(app.y[i])
                app._dynamic_axs[i].set_ylim(auto=True)

            app.dynamic_canvas.draw()        
            
        app._timer = app.dynamic_canvas.new_timer(50)
        app._timer.add_callback(_update_canvas, app)
        app._timer.start()

        app.show()
        app.activateWindow()
        app.raise_()
        qapp.exec_()

    def time_trace_old(self, traces=[0]):
        '''
        Looks like a cardiograph.
        '''

        plt.ion()
        t0 = time.time()
        trace_data = {}
        while True:
            plt.pause(.05)

            y = self.read(100)
            y = 10 * np.log10(y) + 30

            for i, trace in enumerate(traces):

                if trace not in trace_data:
                    trace_data[trace] = {'x': [], 'y': []}
                trace_data[trace]['x'].append(time.time() - t0)
                trace_data[trace]['y'].append(y[trace])

                if len(trace_data[trace]['x']) > 200:
                    trace_data[trace]['x'].pop(0)
                    trace_data[trace]['y'].pop(0)

                plt.subplot(len(traces), 1, i + 1)
                plt.cla()
                line = plt.plot(trace_data[trace]['x'], trace_data[
                                trace]['y'], 'k-', mec='none')
                # plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

            plt.draw()

    def noise_analysis(self):
        r = []
        for i in range(1000):
            r.append(self.read(100))

        import pprint
        pprint.pprint(list(
            np.round(
                np.mean(r, axis=0), 7)
        ))

    def build_v_p_lut(self):
        path = os.path.realpath(os.path.join(
            os.getcwd(), os.path.dirname(__file__)))
        data = np.load(path + '/lightgrid2_cal.npy', allow_pickle=True, encoding='bytes').item()
        vp_lut = {}
        for det in data:
            # 54 db is max dynamic range
            y = data[det][b'meas_ref']
            x = data[det][b'meas_tar']

            x_lb_log = 10 * np.log10(max(x)) - 50  # 54 db max
            x_lb_lin = 10**(x_lb_log / 10)

            xy = [xy for xy in list(zip(x, y)) if xy[0] > x_lb_lin]
            x, y = list(zip(*xy))

            f_spl = interp1d(x, y, kind='linear',
                             bounds_error=False, fill_value=(0., np.nan))

            vp_lut[det] = f_spl

        return vp_lut

    def print_power_status(self, thresh=0.):
        p_tot = 1e6*np.sum(self.read(100))
        print('Total power (%.3f) uW.' % (p_tot))
        if p_tot < thresh:
            print('Recouple (quits when threshold satisfied)!!')

            try:
                while p_tot < thresh:
                    p_tot = 1e6*np.sum(self.read(100)), 'uW'
            except:
                pass

def calibrate_dets():
    from hardware_control.agilent_8153a import Agilent8153A as pm
    pd = pm()
    pds = lightGridWrapper()

    try:
        data = np.load('lightgrid2p1_cal_.npy', allow_pickle=True, encoding='bytes').item()
    except:
        data = {}

    from hardware_control.jdsu_ha9 import JDSUHA9 as atten
    attenuator = atten('GPIB1::10::INSTR')
    attenuator.block_beam(False)

    ndets = 32
    att = np.arange(21, 105, 5)
    for i in [11, 12, 13, 14, 15]:
        input('Connect fiber to detector %d.' % i)

        data[i] = {}

        meas_ref = []
        meas_tar = []
        for a in att:
            attenuator.set_attenuation(a)
            time.sleep(.5)
            pows = np.mean([pds.read(1000)[i] for qq in range(5)])
            meas_tar.append(pows)
            meas_ref.append(pd.read_power(1))

        data[i]['attenuation'] = att
        data[i]['meas_ref'] = meas_ref
        data[i]['meas_tar'] = meas_tar

        plt.subplot(121)
        plt.plot(att, meas_ref, 'r')
        plt.subplot(122)
        plt.plot(att, meas_tar, 'b')
        plt.show()

        np.save('lightgrid2p1_cal', data)

    attenuator.set_attenuation(min(att))
    time.sleep(2)


def calanal():
    from scipy.interpolate import interp1d
    # from mpltools import color
    # color.cycle_cmap(26)

    # allow_pickle=True, encoding='bytes' is the option to load py2 code using py3 program, by
    data = np.load('lightgrid2_cal.npy', allow_pickle=True, encoding='bytes').item()

    for det in data:
        # 54 db is max dynamic range
        y = data[det]['meas_ref']
        x = data[det]['meas_tar']
        f_spl = interp1d(x, y, kind='linear')

        x_lb_log = 10 * np.log10(max(x)) - 54
        x_lb_lin = 10**(x_lb_log / 10)

        xy = [xy for xy in list(zip(x, y)) if xy[0] > x_lb_lin]
        x, y = list(zip(*xy))

        print(f_spl(1.))

        plt.loglog(x, y, '.')
        plt.loglog(x, f_spl(x), '-')
    plt.show()


if __name__ == '__main__':
    # calibrate_dets()
    # calanal()

    x = lightGridWrapper('COM6')
    # x.time_trace(traces=[0, 1, 24, 25])
    # x.noise_analysis()
    x.live_plot()
    # x.time_trace_old([0, 24])
    # x.live_plot([0,1,24,25])
    
    # while True:
    #     time.sleep(.3)
    #     print x.read(100)[25]