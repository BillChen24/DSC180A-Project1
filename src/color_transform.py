import numpy as np

def grayscale(X, filename = "gray"):
    '''
    Change Colors to Gray Scale Images
    '''
    grays = []
    for x in X:
        r, g, b = x[:1024], x[1024:2048], x[2048:]
        gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(int)
        gray = np.concatenate((gray, gray, gray))
        grays.append(gray)
    out = np.array(grays)
    #np.save("data/temp/"+filename, out)
    return out

  
