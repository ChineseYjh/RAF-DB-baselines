"""
    This is the main entry of RAF-DB experiments(supervised learning)
    Usage:
        - Just modify *.conf file, and then run the following command.
        
            python run.py --config_name baseDCNN
"""

from ops import *

if __name__=='__main__':
    try:
        config=parse_config()
        logger=init(config)
        if(config.mode=='train'):
            train(config,logger)
        elif(config.mode=='val'):
            val(config,logger)
        elif(config.mode=='test'):
            if config.classifier=='svm':
                svm_fit_and_test(config,logger)
            else:
                test(config,logger)
    except KeyboardInterrupt:
        logger.error('Stopped by keyboard.\nexit 1.')