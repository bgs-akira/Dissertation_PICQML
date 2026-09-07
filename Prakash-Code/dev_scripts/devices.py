# import datetime
# import serial
# import pyvisa
# import warnings
#
# from prakash.driver.PM100D import PM100D
# from prakash.driver.power_supply import XPOW, XPOWu
# from prakash.driver.optical_switch import LF30CHSM
# import prakash.config as config
#
# # Temporary use this to check that the devices stay the same
# time_stamp = datetime.datetime.now().strftime("%Y-%m-%d(%H-%M-%S.%f)")
#
# powerSupply = None
# opticalPowerMeter = None
# opticalSwitch = None
#
#
# # Power Supply
# try:
#     if config.power_supply == 'XPOW':
#         powerSupply = XPOW(config.power_supply_resource)
#     elif config.power_supply == 'XPOWu':
#         powerSupply = XPOWu(config.power_supply_resource)
#     elif config.power_supply is None:
#         pass
#     else:
#         raise ValueError(f'Power Supply {config.power_supply} not recognized')
#
# except serial.serialutil.SerialException as e:
#     warnings.warn(f"Error when opening power supply: {e}")
#
#
# # Optical Power Meter
# try:
#     if config.optical_power_meter == 'PM100D':
#         opticalPowerMeter= PM100D(config.optical_power_meter_resource)
#         opticalPowerMeter.set_wav(1550)
#
#     elif config.optical_power_meter is None:
#         pass
#     else:
#         raise ValueError(f'Optical Power Meter {config.optical_power_meter} not recognized')
#
# except pyvisa.errors.VisaIOError as e:
#     warnings.warn(f"Error when opening power meter: {e}")
#
# # Optical Switch
# try:
#     if config.optical_switch == 'LF30CHSM':
#         opticalSwitch = LF30CHSM(config.optical_switch_resource)
#     else:
#         raise ValueError(f'Optical Switch {config.optical_switch} not recognized')
#
# except serial.serialutil.SerialException as e:
#     warnings.warn(f"Error when opening optical switch: {e}")