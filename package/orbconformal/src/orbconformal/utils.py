import re
import numpy as np

def check_character_percent(x, name = "x"):
    """
    check if string is a percentage (and return that as proportion if so)

    Parameters
    ----------
    x : string
        element to examine
    name : string
        string of name of x (assumingly this is used inside a function, that
        may not call it "x")

    Returns
    -------
    float
        the proportion version of the percentage (if string was a percentage
        and meets other expectations)
    """
    assert len(re.findall("%$", x)) == 1, \
        f"if {name!r} is a character it must be '__%'."

    percentage = float(re.sub("%","", x))/100

    assert percentage <= 1 and percentage > 0, \
        f"if {name!r} is entered as a percent, " +\
        "it must be a percentage <= 100% and " +\
        "greater than 0%"

    return percentage
