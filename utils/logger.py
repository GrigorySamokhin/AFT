"""
    Routines for logging subsystem initialization.
"""

__all__ = ['initialize_logging']

import os
import sys
import logging
import argparse
from typing import Tuple, Optional
from .string import split_str, pretty_print_dict_to_str
from .envs import get_env_stats


def prepare_logger(logging_dir_path: Optional[str],
                   logging_file_name: Optional[str]) -> Tuple[logging.Logger, bool]:
    """
    Prepare logger.

    Parameters
    ----------
    logging_dir_path : str or None
        Path to logging directory.
    logging_file_name : str or None
        Name of logging file.

    Returns
    -------
    logging.Logger
        Logger instance.
    bool
        Whether the logging file already exists.
    """
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_exist = False
    if (logging_dir_path is not None) and logging_dir_path and (logging_file_name is not None) and logging_file_name:
        log_file_path = os.path.join(logging_dir_path, logging_file_name)
        if not os.path.exists(logging_dir_path):
            os.makedirs(logging_dir_path, exist_ok=True)
            log_file_exist = False
        else:
            log_file_exist = (os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0)
        fh = logging.FileHandler(log_file_path)
        logger.addHandler(fh)
        if log_file_exist:
            logging.info("--------------------------------")
    return logger, log_file_exist


def initialize_logging(logging_dir_path: Optional[str] = None,
                       logging_file_name: Optional[str] = None,
                       main_script_path: Optional[str] = None,
                       script_args: Optional[argparse.Namespace] = None,
                       packages: Optional[str] = "log_packages",
                       pip_packages: Optional[str] = "log_pip_packages",
                       check_ffmpeg: bool = False) -> Tuple[logging.Logger, bool]:
    """
    Initialize logging subsystem.

    Parameters
    ----------
    logging_dir_path : str or None, default None
        Path to logging directory.
    logging_file_name : str or None, default None
        Name of logging file.
    main_script_path : str or None, default None
        Path to main running script.
    script_args : argparse.Namespace or None, default None
        Script arguments.
    packages : str or None, default 'log_packages'
        Name of field on script_args for package name list to inspect only __version__.
    pip_packages : str or None, default 'log_pip_packages'
        Name of field on script_args for package name list to inspect by 'pip show'.
    check_ffmpeg : bool, default False
        Whether to show FFmpeg version.

    Returns
    -------
    logging.Logger
        Logger instance.
    bool
        Whether the logging file already exists.
    """
    logger, log_file_exist = prepare_logger(
        logging_dir_path=logging_dir_path,
        logging_file_name=logging_file_name)
    logging.info("Script command line:\n{}".format(" ".join(sys.argv)))
    if script_args is not None:
        script_arg_dict = script_args.__dict__
        script_arg_dict = {k: script_arg_dict[k] for k in sorted(script_arg_dict.keys())}
        logging.info("Script arguments:\n{}".format(pretty_print_dict_to_str(script_arg_dict)))
        if packages is not None:
            packages = split_str(script_arg_dict[packages]) if (packages in script_arg_dict.keys()) else []
        if pip_packages is not None:
            pip_packages = split_str(script_arg_dict[pip_packages]) if (pip_packages in script_arg_dict.keys()) else []
    else:
        packages = None
        pip_packages = None
    logging.info("Environment statistics:\n{}".format(pretty_print_dict_to_str(get_env_stats(
        packages=packages,
        pip_packages=pip_packages,
        main_script_path=main_script_path,
        check_ffmpeg=check_ffmpeg))))
    return logger, log_file_exist
