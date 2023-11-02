# import scipy.io
# mat = scipy.io.loadmat('data/syl1.mat')
# import pandas as pd
# print(mat)
import os
from mat4py import loadmat

data = loadmat('data/phn1.mat')
# print(data)

# duration = []
# sylStart = [0]
# for i in range(len(data['spurtSylTimes'])):
#     duration.append(data['spurtSylTimes'][i][1] - data['spurtSylTimes'][i][0])
#     sylStart.append(data['spurtSylTimes'][i][0])
# data['spurtSylTimes'] = duration
# data['sylStart'] = sylStart
# print(data)



duration = []
phnStart = [0]
for i in range(len(data['phnTimes'])):
    duration.append(round(data['phnTimes'][i][1] - data['phnTimes'][i][0],2))
    phnStart.append(data['phnTimes'][i][0])
data['phnTimes'] = duration
data['phnStart'] = phnStart
print(data)
# new = pd.DataFrame.from_dict(data)

# print(new)

# linguistic = ["phn","syl","wav"]
# folder = os.listdir('data/phn')

# print(folder)

# for files in folder:
#     # cur_folder = os.listdir(folder + "/")
#     print(files)
## populating sample frequency
# out = []
# count=0
# for i in range(500):
#     if data['sylStart'][count]*1000 == i:
#         count+=1
#         out.append(1)
#     else: out.append(0)
# print(out)

## population phoneme duration

out_ph = []
count_ph = 0

for i in range(500):
    if data['phnStart'][count_ph]*1000 == i:
        count_ph+=1

    else:
        out_ph.append(data['phnTimes'][count_ph-1]*1000)

print(out_ph)