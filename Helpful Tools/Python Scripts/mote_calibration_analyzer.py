import os
import requests
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Go to inspect element, on the machine you are interested in, and choose the something like "37"
# Use https://curl.trillworks.com/ to convert

cookies = {
    '_ga': 'GA1.2.354410958.1484069253',
    'mp_6236728f0c61399bdb15b5a17d1fbf1c_mixpanel': '^%^7B^%^22distinct_id^%^22^%^3A^%^20^%^22simon^%^40petasense.com^%^22^%^2C^%^22^%^24initial_referrer^%^22^%^3A^%^20^%^22^%^24direct^%^22^%^2C^%^22^%^24initial_referring_domain^%^22^%^3A^%^20^%^22^%^24direct^%^22^%^7D',
    'session': '.eJw9z8FugkAUheFXaWbtYgDZmHQhGSSQ3ku0g5PLhlAchUE0ESwyxnevuuj-nC_576zYX3Rfs8W-PPZ6xopmxxZ39vHDFgxFzHOR1NDFFkQ2kVo1ZDYNyOWYRklHLlmwiUEbHNHWLZqWozhwVNiixDaVaz8VIQe76UBmN5K7FlRiQK1HdEOf3Py5iTnJl4QGI5qjXTq5iH2wtckFjSiCOo1oyjuYp9G2AbWtwYQ3NOSRCQypzCMJn-wxY2U1NL-6KKvqfD0N7xJnxnrd9835VLR6-i8DuTqCC97zPYCoJvzmnOzS-1Khh1E4pKJywNII5jCCjP2Xfu315W0yx5-zxx96eGZZ.DbEy_A.97UojoCmGcSI1iqrB-pvuxgfx00',
}

headers = {
    'Pragma': 'no-cache',
    'Origin': 'https://mfg.petasense.com',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://mfg.petasense.com/',
    'Connection': 'keep-alive',
    'Cache-Control': 'no-cache',
}

# Get the list of device ids
file = open("data/P12-IDs.txt", 'r')
device_ids = file.read().split("\n")
num_ids = len(device_ids)
num_meas = 15
acc_832m1_x = np.zeros((num_ids, num_meas))
acc_832m1_y = np.zeros((num_ids, num_meas))
acc_832m1_z = np.zeros((num_ids, num_meas))

# Get calibration info
for i in range(num_ids):
    request_url = 'https://api.petasense.com/mfgapp/calibration/report?device_id=' + str(device_ids[i])
    calibration_info = requests.get(request_url, headers=headers, cookies=cookies)
    calibration_info = calibration_info.json()

    for j in range(num_meas):
        acc_832m1_x[i, j] = calibration_info['data_lsm6ds3']['rms']['x'][j]
        acc_832m1_y[i, j] = calibration_info['data_lsm6ds3']['rms']['y'][j]
        acc_832m1_z[i, j] = calibration_info['data_lsm6ds3']['rms']['z'][j]

    print(i, "files complete")

np.savetxt('data/acc_lsm_x.csv', acc_832m1_x, fmt='%.18e', delimiter=',')
np.savetxt('data/acc_lsm_y.csv', acc_832m1_y, fmt='%.18e', delimiter=',')
np.savetxt('data/acc_lsm_z.csv', acc_832m1_z, fmt='%.18e', delimiter=',')

avg_x = np.zeros(num_meas); avg_y = np.zeros(num_meas); avg_z = np.zeros(num_meas)
med_x = np.zeros(num_meas); med_y = np.zeros(num_meas); med_z = np.zeros(num_meas)
min_x = np.zeros(num_meas); min_y = np.zeros(num_meas); min_z = np.zeros(num_meas)
max_x = np.zeros(num_meas); max_y = np.zeros(num_meas); max_z = np.zeros(num_meas)

for i in range(num_meas):
    avg_x[i] = np.average(acc_832m1_x[i, :])
    avg_y[i] = np.average(acc_832m1_y[i, :])
    avg_z[i] = np.average(acc_832m1_z[i, :])
    med_x[i] = np.median(acc_832m1_x[i, :])
    med_y[i] = np.median(acc_832m1_y[i, :])
    med_z[i] = np.median(acc_832m1_z[i, :])
    min_x[i] = np.min(acc_832m1_x[i, :])
    min_y[i] = np.min(acc_832m1_y[i, :])
    min_z[i] = np.min(acc_832m1_z[i, :])
    max_x[i] = np.max(acc_832m1_x[i, :])
    max_y[i] = np.max(acc_832m1_y[i, :])
    max_z[i] = np.max(acc_832m1_z[i, :])


plt.plot(avg_x, 'b')
plt.plot(med_x, 'k')
plt.plot(min_x, 'r')
plt.plot(max_x, 'g')
plt.show()

plt.plot(avg_y, 'b')
plt.plot(med_y, 'k')
plt.plot(min_y, 'r')
plt.plot(max_y, 'g')
plt.show()

plt.plot(avg_z, 'b')
plt.plot(med_z, 'k')
plt.plot(min_z, 'r')
plt.plot(max_z, 'g')
plt.show()