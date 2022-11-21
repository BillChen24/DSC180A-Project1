import numpy as np

def grayscale(X, filename = "gray"):
    grays = []
    for x in X:
        r, g, b = x[:1024], x[1024:2048], x[2048:]
        gray = np.concatenate((0.2989 * r, 0.5870 * g, 0.1140 * b))
        grays.append(gray)
    out = np.array(grays)
    #np.save("data/temp/"+filename, out)
    return out
