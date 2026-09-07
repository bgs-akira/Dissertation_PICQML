import datetime
from pathlib import Path

# from prakash.utils import LogUtils

time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")

# power_supply = 'XPOWu'
# power_supply_resource = ['COM12', 'COM13', 'COM14']
#
# optical_power_meter = 'PM100D'
# # Find resource by pyvisa.ResourceManager().list_resources()
# optical_power_meter_resource = 'USB0::0x1313::0x8078::P0022518::INSTR'
#
# optical_switch = 'LF30CHSM'  # LFIBER 30 Channel SM fibre
# optical_switch_resource = 'COM4'

# config logging
# LogUtils.log_config(time_stamp)

# directory
# home_dir = r'C:\Users\PhotonsLocalAdmin\Documents\prakash_one_control\prakash'
home_dir = Path(__file__).resolve().parent.__str__()