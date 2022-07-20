"""
    Routines for system environment processing.
"""

__all__ = ['get_env_stats']

import os
import sys
import subprocess
import platform
from typing import List, Dict, Optional


def get_platform_info() -> Dict[str, str]:
    """
    Get platform (operational system) information.

    Returns
    -------
    dict(str, str)
        Resulted info.
    """
    platform_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "libc_ver": platform.libc_ver(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
    }
    return platform_info


def get_sys_python_version() -> str:
    """
    Get python version.

    Returns
    -------
    str
        Resulted info.
    """
    try:
        return "{0}.{1}.{2}".format(*sys.version_info[0:3])
    except Exception:
        return "unknown"


def get_git_commit_info(script_path: Optional[str] = None) -> str:
    """
    Get the last Git repo commit information.

    Parameters
    ----------
    script_path : str or None, default None
        Path to a script from repository.

    Returns
    -------
    str
        Resulted info.
    """
    try:
        if (script_path is not None) and script_path:
            pwd = os.path.dirname(os.path.realpath(script_path))
        else:
            pwd = os.getcwd()
        if os.name == "nt":
            command = "cmd /V /C \"cd {} && git log -n 1\"".format(pwd)
        else:
            command = ["cd {}; git log -n 1".format(pwd)]
        output_bytes = subprocess.check_output(command, shell=True)
        output_text = output_bytes.decode("utf-8").strip()
    except Exception:
        output_text = "unknown"
    return output_text


def get_package_versions(module_names: List[str]) -> Dict[str, str]:
    """
    Get packages information by inspecting __version__ attribute.

    Parameters
    ----------
    module_names : list(str)
        List of module names.

    Returns
    -------
    dict(str, str)
        Dictionary with module versions.
    """
    module_versions = {}
    for module_name in module_names:
        try:
            module_versions[module_name] = __import__(module_name).__version__
        except ImportError:
            module_versions[module_name] = None
        except AttributeError:
            module_versions[module_name] = "unknown"
    return module_versions


def get_pip_package_descriptions(package_names: List[str],
                                 python_version: str = "") -> Dict[str, str]:
    """
    Get packages information by using 'pip show' command.

    Parameters
    ----------
    package_names : list(str)
        List of package names.
    python_version : str, default ''
        Python version ('2', '3', '') appended to 'pip' command.

    Returns
    -------
    dict(str, str)
        Dictionary with package descriptions.
    """
    package_descriptions = {}
    for package_name in package_names:
        try:
            output_bytes = subprocess.check_output([
                "pip{0}".format(python_version),
                "show", package_name])
            output_text = output_bytes.decode("utf-8").strip()
        except (subprocess.CalledProcessError, OSError):
            output_text = "unknown"
        package_descriptions[package_name] = output_text
    return package_descriptions


def get_ffmpeg_version() -> str:
    """
    Get FFmpeg version.

    Returns
    -------
    str
        Resulted info.
    """
    try:
        output_bytes = subprocess.check_output(["ffmpeg", "-version"])
        output_text = output_bytes.decode("utf-8").strip().split("\n")[0]
    except (subprocess.CalledProcessError, OSError):
        output_text = "unknown"
    return output_text


def get_env_stats(packages: Optional[List[str]],
                  pip_packages: Optional[List[str]],
                  main_script_path: Optional[str] = None,
                  check_ffmpeg: bool = False) -> Dict[str, str]:
    """
    Get environment statistics.

    Parameters
    ----------
    packages : list(str) or None
        list of package names to inspect only __version__.
    pip_packages : list(str) or None
        List of package names to inspect by 'pip show'.
    main_script_path : str or None, default None
        Path to main running script.
    check_ffmpeg : bool, default False
        Whether to show FFmpeg version.

    Returns
    -------
    dict(str, str)
        Resulted string with information.
    """
    env_stat_dict = {
        "platform": get_platform_info(),
        "cwd": os.getcwd(),
        "sys_python_version": get_sys_python_version(),
        "git_commit": get_git_commit_info(main_script_path),
    }
    if check_ffmpeg:
        env_stat_dict["ffmpeg"] = get_ffmpeg_version()
    if (packages is not None) and (len(packages) > 0):
        module_versions = get_package_versions(packages)
        env_stat_dict["packages"] = module_versions
    if (pip_packages is not None) and (len(pip_packages) > 0):
        python_version = str(sys.version_info[0])
        package_descriptions = get_pip_package_descriptions(packages, python_version)
        env_stat_dict["pip_packages"] = package_descriptions
    return env_stat_dict
