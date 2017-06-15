import logging

def config_logger(logfn='', mode='a'):
    """
    config logger
    """
    s_fmt= '%(asctime)s@%(filename)s:%(lineno)d:%(process)d:%(levelname)s: %(message)s'
    # formatter
    log_fmt = logging.Formatter(s_fmt)
    # root logger
    logger = logging.getLogger()
    # level
    logger.setLevel(logging.INFO)
    # handlers
    s_hdr = logging.StreamHandler()
    s_hdr.setFormatter(log_fmt)
    logger.addHandler(s_hdr)
    if logfn:
        f_hdr = logging.FileHandler(logfn, mode=mode)
        f_hdr.setFormatter(log_fmt)
        logger.addHandler(f_hdr)

