import numpy as np
import ot

def sample_color(X_source, X_target, n):
    sample1 = np.random.randint(X_source.shape[0], size = n)
    sample2 = np.random.randint(X_target.shape[0], size = n)
    return X_source[sample1], X_target[sample2]

def color_ot_build(X_source, X_target, ot_type = ot.da.EMDTransport()):
    ot_type.fit(Xs=Xs, Xt=Xt)
    return ot_type

def color_ot_transform(X_test_source, ot_type):
    X_test_ot = ot_type.transform(X_test_source)
    return X_test_ot
