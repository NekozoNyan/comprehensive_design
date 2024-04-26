import numpy as np

w, h = 224, 224
a = [0.343750, 0.908750, 0.156162, 0.650047]
a = (a * np.array([w,h,w,h])).astype(np.uint16).tolist()
print(a)