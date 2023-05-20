# import scipy.io
# mat = scipy.io.loadmat('data/syl1.mat')
import pandas as pd
# print(mat)
import os
from mat4py import loadmat

data = loadmat('data/phn1.mat')
print(data)

# duration = []
# sylStart = []
# for i in range(len(data['spurtSylTimes'])):
#     duration.append(data['spurtSylTimes'][i][1] - data['spurtSylTimes'][i][0])
#     sylStart.append(data['spurtSylTimes'][i][0])
# data['spurtSylTimes'] = duration
# data['sylStart'] = sylStart
# print(data)

# new = pd.DataFrame.from_dict(data)

# print(new)

# linguistic = ["phn","syl","wav"]
# folder = os.listdir('data/phn')

# print(folder)

# for files in folder:
#     # cur_folder = os.listdir(folder + "/")
#     print(files)
