import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime

# calibration data structure
cdt = np.dtype([
    ('pins', np.uint8),
    ('addrs', np.uint8, 2), # address of each MZI
    ('paras', np.float32, 4), # parameters of electrical power vs. optical phase fitting function
    ('time', np.datetime64) # calibration operated time
])

# CaliData = {}
# CaliData['rising_time'] = 0.1 
# CaliData['width'] = 6
# CaliData['depth'] = 6
# CaliData['rising_time'] = 0.1 
# CaliData['phase_func']=np.array([],dtype=(100,cdt))

def fit_func(x, a, b, c, d):
    return a*np.sin(x*b+c)+d

class PhaseShifter(object):
    """
    Phase shifter
    """
    def __init__(self, addr, pin, rising_time=0.01, calidata=None):
        
    
        self.addr = addr
        self.pin = pin
        if calidata is None:
            self.paras = None
        else:
            self.paras = calidata['paras'][np.where(calidata['pins']==pin)]

        self.power_read = None
        self.current_set = None
        # self.volts = np.linspace(0,10,100)
        # self.intensity = None        
        # self.func = None
        
    def SweepIV(self, ps, v_max=10, v_min=0, num=10):
        vv = np.linspace(v_min, v_max, num)
        ii = np.zeros_like(vv)
        for i, v in enumerate(vv):
            ps.v[self.pin] = v
            ii[i] = ps.i[self.pin] 
        return [vv, ii]
    
    def SweepVoltPhase(self, ps, opm, v_max=10, v_min=0, num=30):
        volts = np.sqrt(np.linspace(v_min**2, v_max**2, num))
        op = np.zeros_like(volts)
        for i, v in enumerate(volts):
            ps.v[self.pin] = v
            op[i] = opm.read()
        return volts, op
    
    def SweepCurrPhase(self, ps, opm, i_max=10, i_min=0, num=30):
        currs = np.sqrt(np.linspace(i_min**2, i_max**2, num))
        op = np.zeros_like(currs)
        for i, c in enumerate(currs):
            ps.i[self.pin] = c
            op[i] = opm.read()
        return currs, op
    
    def SweepFitPhaseDummy(self, i_max=10, i_min=0, num=30, plot=False):
        currs = np.sqrt(np.linspace(i_min**2, i_max**2, num))
        volts = currs*0.1
        pp = currs*volts

        paras = np.random.normal([1, 1, 0.1, 1], [.1, .1, .01, .1])
        rms = paras[0]*0.01
        op = fit_func(pp, *paras) + np.random.normal(0, rms, size=30)

        popt, pcov = curve_fit(fit_func, pp, op)
        self.paras = popt
        if plot is True:
            plt.plot(pp, op, 'r*')
            plt.plot(pp, fit_func(pp, *popt))
            plt.show()
        return None
    
    def SweepFitPhase(self, ps, opm, i_max=10, i_min=0, num=30):
        currs = np.sqrt(np.linspace(i_min**2, i_max**2, num))
        volts = np.zeros_like(currs)
        op = np.zeros_like(currs)
        for i, c in enumerate(currs):
            ps.i[self.pin] = c
            volts[i] = ps.v[self.pin]
            op[i] = opm.read()
        pp = currs*volts
        popt, pcov = curve_fit(fit_func, pp, op)
        self.paras = popt
        return popt
    
    def SaveCali(self, calidata):
        calidata['paras'][np.where(calidata['pins']==self.pin)] = self.paras
        return None

class Clements(object):

    def __init__(self, N):
        self.N = N
        self.addrs = []
        for xx in range(self.N):
            for yy in range(xx%2, self.N-1, 2):
                self.addrs.append([xx, yy])
        
        chains = []
        for i in range(1, self.N):
            if i%2 ==1:
                chains.append([(j,-j+i-1) for j in range(i)])
            else:
                chains.append([(self.N-1-j, self.N-1+j-i ) for j in range(i)])   
        self.clements_list = chains
        
    def Route(self, dev_addr):
        """
        Route the output port for a given phaseshifter
        """
        assert tuple(dev_addr) in self.addrs
        xx = range(self.N)
        
        # plot two guiding function
        y1 = [ x - dev_addr[0] + dev_addr[1] for x in range(self.N) ]
        y2 = [ -x + dev_addr[0] + dev_addr[1] for x in range(self.N) ]        
        # get two parametric func
        y1 = [ -2 - y if y <-1 else 2*self.N-2 - y if y > self.N-1 else y for y in y1]
        y2 = [ -2 - y if y <-1 else 2*self.N-2 - y if y > self.N-1 else y for y in y2] 
        # reflect the func 
        r1 = [(x,y) for x,y in zip(xx, y1)]
        r2 = [(x,y) for x,y in zip(xx, y2)]

        # determine orientation
        port_in = [ y[0] if y[1] - y[0] == 1 else y[0] + 1 for y in [y1, y2]]
        port_out = [ y[-1] + 1 if y[-1] - y[-2] == 1 else y[-1]  for y in [y1, y2]]
        # yield waveguide number to avoid N and -1
        port_in = [ self.N-1 if i == self.N else 0 if i == -1 else i for i in port_in]
        port_out = [ self.N-1 if i == self.N else 0 if i == -1 else i for i in port_out]

        in1 = [ (x,y) for x,y in r1 if y != self.N and y != -1 and x < dev_addr[0]]
        in2 = [ (x,y) for x,y in r2 if y != self.N and y != -1 and x < dev_addr[0]]
        out1 = [ (x,y) for x,y in r1 if y != self.N and y != -1 and x > dev_addr[0]]
        out2 = [ (x,y) for x,y in r2 if y != self.N and y != -1 and x > dev_addr[0]]
        return in1, in2, out1, out2, port_in, port_out
    
    # def RouteExt(self, dev_addr):
    #     MZILeft = [dev_addr[0]-1, dev_addr[1]-1]
    #     MZIRight = [dev_addr[0]+1, dev_addr[1]-1]
    #     in1, in2, _, _, port_in, _  = self.Route(MZILeft)
    #     _, _, out1, out2, _, port_out = self.Route(MZIRight)
    #     return in1, in2, out1, out2, port_in, port_out

class Calibration(object):

    def __init__(self, mesh, calidata=None):
        assert type(mesh) == Clements
        self.mesh = mesh
        self.n_pins = int((mesh.N**2 - mesh.N)/2)
        
        if calidata == None:
            calidata = np.zeros(self.n_pins, dtype=cdt)
            calidata['addrs'] = mesh.addrs
            calidata['pins'] = np.arange(self.n_pins)
        self.calidata = calidata
        self.pins = calidata['pins']
        
    def CaliExt(self):
        cali_chains = self.mesh.clements_list[::-2] +  self.mesh.clements_list[-2::-2]
        for c in cali_chains:
            for a in c:
                pin = self.calidata['pins'][np.where(np.all(self.calidata['addrs'][:]==a,axis=1))]
                ps = PhaseShifter(addr=a, pin=pin)
                ps.SweepFitPhaseDummy()
                ps.SaveCali(self.calidata)
        return None

if __name__ == '__main__':
    # ps1 = PhaseShifter(pin=1, addr=(0,0), calidata=calidata)
    # print(ps1.SweepFitPhaseDummy(plot=True))

    M = Clements(10)
    cali = Calibration(M)
    print(cali.CaliExt())
    
    