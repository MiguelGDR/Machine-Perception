from matplotlib import pyplot as plt
import numpy as np

# focal length (cm)
f = 0.25 
# stereo system distance (cm)
T = 12

p = np.linspace(100,2800,55)

dis =  f * T / p

plt.plot(p,dis)
plt.xlabel('distances in cm')
plt.ylabel('disparity')

plt.show()