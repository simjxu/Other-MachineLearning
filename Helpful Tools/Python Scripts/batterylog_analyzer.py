import numpy as np
import csv
import matplotlib.pyplot as plt

csvfile = open('data/TX6_batterylog.csv')
fieldnames = ['timestamp', 'event', 'event start', 'reason', 'battery volt', 'rssi', 'temperature', 'payload length']
reader = csv.DictReader(csvfile, fieldnames)

# Info matrix
# col 1: total reporting interval
# col 2: cold boot and multipart upload
numcols = 2
datamat = np.zeros((0,numcols))

# next(reader)             # skip the header line
indx = 0
flag = 0
cold = 0
for row in reader:

    # block start
    if row['reason'] == 'get_config' and flag == 0:
        startidx = indx
        flag = 1
        starttime = int(row['event start'])/1000000.0

    elif row['reason'] == 'cold_boot' and cold == 0:
        coldidx = indx
        cold = 1
        coldtime = int(row['event start'])/1000000.0

    elif row['reason'] == 'multipart_upload_samples offset:0' and cold == 1:
        uploadidx = indx
        uploadtime = int(row['event start']) / 1000000.0
        cold = 0

    # block end, either upload battery test log or ready to sleep is sufficient
    elif flag == 1 and row['reason'] == 'upload_battery_test_log':
        endidx = indx
        endtime = int(row['event start'])/1000000.0
        newrow = np.matrix((endtime-starttime, uploadtime-coldtime))
        datamat = np.append(datamat, newrow, 0)
        flag = 0





    indx += 1


print(datamat.shape)
print("average connecting time:", np.mean(datamat[:,1]))
print("total connecting time:", np.sum(datamat[:,1]))
print("average transmit time:", np.mean(datamat[:,0]))
print("total transmit time:", np.sum(datamat[:,0]))
plt.plot(datamat[:,0])
plt.title('transmit time')
plt.show()











# print(row['event start'])

# set config, upload_battery_test_log | ready to sleep

# Get time between cold_boot and multipart_upload_samples offset:0
# If 0, ignore

# with open('data/30000c2a690cc8e7_logs.csv') as f:
#     reader = csv.reader(f)
#     next(reader)            # skip the header line
#     for row in reader:
#         print(int(row[2]))
