import numpy as np
import pandas as pd
import os

import prakash.config as config
from prakash.components import Phaseshifter, ExPhaseshifter, MZI
from prakash.utils import DFUtils

# The key requirement of Mesh is to be able to call mesh.decompose(U). This is the class to control the whole chip-package.
# Store an actual mesh of mzis and external phaseshifters in memory, to avoid reading data files every time an mzi is called.
class Mesh(object):
    """
    Class for the chip mesh.
    """

    def __init__(self, name='prakash_one', power_supply=None, mapping=None):
        self.name = name

        if mapping is None:
            mapping = {}
            df = pd.read_csv(config.home_dir + rf'\{name}\phaseshifters.csv')
            for id in range(len(df)):
                label = (df['i'][id], df['j'][id])
                if label in mapping.keys():
                    raise Exception(f'Duplicate phaseshifter label {label}')
                else:
                    mapping[label] = dict(df.iloc[id])
        self.mapping = mapping  # phaseshifter and power_supply channel mapping

        self.power_supply = power_supply

        self._mzis = {}
        for i in range(9):
            for j in range(10):
                if (i + j) % 2 == 1:
                    continue
                else:
                    mzi_label = (i, j)
                    channel1 = self.get_channel((i, j))
                    channel2 = self.get_channel((i + 1, j))
                    self._mzis[mzi_label] = MZI(name, mzi_label, power_supply, channels=(channel1, channel2))

        self._ex_ps = {}
        for i in [0, 9]:
            for j in [1, 3, 5, 7, 9]:
                ps_label = (i, j)
                channel = self.get_channel(ps_label)
                self._ex_ps[ps_label] = ExPhaseshifter(name, ps_label, power_supply, channel=channel)

    def backup_calibration(self, file_dir=None):
        import shutil

        if file_dir is None:
            file_dir = config.home_dir + rf'\{self.name}\backup_params\{config.time_stamp}'
            os.makedirs(file_dir, exist_ok=True)

        # mzi parameters
        for mzi_label in self._mzis.keys():
            targetMZI = self._mzis[mzi_label]
            targetMZI.save_params(file=DFUtils.create_filename(file_dir + rf'\mzi_params\{mzi_label}.json'))

        # external phaseshifter parameters
        for ps_label in self._ex_ps.keys():
            targetPS = self._ex_ps[ps_label]
            targetPS.save_params(file=DFUtils.create_filename(file_dir + rf'\ps_params\{ps_label}.json'))

        # phaseshifter sweep data for interpolation
        shutil.copytree(config.home_dir + rf'\{self.name}\ps_sweep', file_dir + rf'\ps_sweep')

        # sigma sweep raw data
        shutil.copytree(config.home_dir + rf'\{self.name}\sigma_sweep', file_dir + rf'\sigma_sweep')


    def save_mapping(self):
        """Save phaseshifter to power supply channel mapping. """
        ps_list = list(self.mapping.values())
        df = pd.DataFrame(ps_list)

        df.to_csv(config.home_dir + rf'\{self.name}\phaseshifters.csv', index=False)

    def update_power_supply(self, power_supply):
        self.power_supply = power_supply

    def get_channel(self, ps_label):
        return self.mapping[ps_label]['Channel']

    def mzi(self, mzi_label):
        return self._mzis[mzi_label]

    def phaseshifter(self, ps_label):
        if ps_label in self._ex_ps.keys():
            targetPS = self._ex_ps[ps_label]
        else:
            raise ValueError(rf'Phaseshifter {ps_label} not an external phaseshifter')

        return targetPS

    def set_mzi(self, mzi_labels, *args, **kwargs):
        mzi_labels = np.atleast_2d(mzi_labels)

        for mzi_label in mzi_labels:
            targetMZI = self.mzi(tuple(mzi_label))
            targetMZI.set_phase(*args, **kwargs)

    def set_phaseshifter(self, ps_labels, *args, **kwargs):
        ps_labels = np.atleast_2d(ps_labels)

        for ps_label in ps_labels:
            targetPS = self.phaseshifter(tuple(ps_label))
            targetPS.set_phase(*args, **kwargs)

    @staticmethod
    def get_diagonal(k):
        """Return the MZI and External Phaseshifter labels along diagonal k, in order from mode 0 to mode 9"""
        if k < 0 or k > 9:
            raise ValueError(f'Diagonal k={k} out of range')

        components = {'mzi': [], 'ps': []}

        for i in range(min(2 * k + 1, 9)):
            j = 2 * k - i
            if j < 0 or j > 9:
                continue
            else:
                components['mzi'].append((i, j))

        if k >= 5:
            components['ps'].append((9, 2 * k - 9))
        else:
            components['ps'].append((0, 2 * k + 1))

        return components

    def zero_mzi(self, mzi_labels):
        mzi_labels = np.atleast_2d(mzi_labels)

        for mzi_label in mzi_labels:
            targetMZI = self.mzi(tuple(mzi_label))
            targetMZI.zero()

    def zero_all(self):
        self.power_supply.zero_all()

class PartialMesh(Mesh):
    # TODO: finish this class. Enable partial control of the mesh. And maybe also the option of
    #  re-labelling the phaseshifters.
    pass
