"""
    String processing auxiliary functions.
"""

__all__ = ['split_str', 'pretty_print_dict_to_str']

import json
from typing import List, Dict, Optional


def split_str(value: Optional[str],
              sep: str = ",") -> Optional[List[str]]:
    """
    Split string into list of strings.

    Parameters
    ----------
    value : str or None
        Processed string.
    sep : str, default ','
        Item seperator.

    Returns
    -------
    list(str) or None
        Resulted list of strings.
    """
    if value is None:
        return None
    value_list = value.replace(" ", "").split(sep)
    assert (len(value_list) > 0)
    if not value_list[0]:
        return []
    return value_list


def pretty_print_dict_to_str(d: Dict[str, str]) -> str:
    """
    Pretty print of dictionary d to json-formatted string.

    Parameters
    ----------
    d : dict(str, str)
        Processed dictionary.

    Returns
    -------
    str
        Resulted string.
    """
    out_text = json.dumps(d, indent=4)
    return out_text
