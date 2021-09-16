import numpy as np
import time
import orbconformal as oc

def test_stamp_area():
    """
    test stamp_area2 - checking memoization only

    Note this will only work the first time a value
    """
    for lat in np.random.uniform(size = 100)*100:

        s_time = time.time()
        oc.stamp_area(lat)
        elapsed1 = time.time() - s_time


        s_time2 = time.time()
        oc.stamp_area(lat)
        elapsed2 = time.time() - s_time2

        assert elapsed1 > elapsed2,\
            "expect meomization to kick in (%2.2f)" % lat
