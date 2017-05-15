import numpy as np
import matplotlib.pyplot as plt

train = np.load('reinforcetraindata.npy')
# filt = [i if train[i][0] is None else continue for i in range(len(train))]

ex = train[15000][0]['vision']
ex = np.mean(ex, axis=2)
view_mask = np.where(ex == 0)


plt.imshow(ex)
plt.show()

# filt = []
# for i in range(len(train)):
#     if train[i][0] is None:
#         filt.append(i)
#
# train = np.delete(train, filt)

def find_idx(in_img):
    
    for i in range(in_img):
