import orbconformal as oc
import numpy as np


def test_check_character_precent():
    assert oc.check_character_percent("100%", "x") == 1, \
        "expect 100% to translate to 1.0"

    assert np.abs(oc.check_character_percent("99.99%", "x") - .9999) < 1e-07, \
        "expect 99.99% to translate to .9999"

    correct_error0 = False
    try:
        oc.check_character_percent("0%", "x")
    except AssertionError:
        correct_error0 = True

    assert correct_error0, \
        "0% should error in check_character_percent, but didn't "+\
        "(or didn't with an AssertionError)"


    correct_error_double_point = False
    try:
        oc.check_character_percent(".99.99%", "x")
    except ValueError:
        correct_error_double_point = True

    assert correct_error0, \
        ".99.99% should error in check_character_percent, but didn't "+\
        "(or didn't with an ValueError)"
