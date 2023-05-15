# import scipy.io
# mat = scipy.io.loadmat('data/syl1.mat')

# print(mat)

from mat4py import loadmat

data = loadmat('data/phn1.mat')
print(data)