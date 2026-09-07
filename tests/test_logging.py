import logging

from prakash.utils import LogUtils
import prakash.config as config


LogUtils.log_config(config.time_stamp, log_dir=r'..\Results\logs\test_logs', filename='test_log_1')
logging.info(f'{config.time_stamp}')
logging.info('This is test log 1')

LogUtils.log_config(config.time_stamp, log_dir=r'..\Results\logs\test_logs', filename='test_log_2')
logging.info(f'{config.time_stamp}')
logging.info('This is test log 2')

