# coding: utf-8
import os
import requests
import json
from datetime import datetime
import numpy as np

# Set cookies and headers
# Update these after logging in to app.petasense.com
# Go to inspect element, on the machine you are interested in, and choose the something like "37"
# Use https://curl.trillworks.com/ to convert
cookies = {
    '_ga': 'GA1.2.354410958.1484069253',
    'mp_6236728f0c61399bdb15b5a17d1fbf1c_mixpanel': '^%^7B^%^22distinct_id^%^22^%^3A^%^20^%^22simon^%^40petasense.com^%^22^%^2C^%^22^%^24initial_referrer^%^22^%^3A^%^20^%^22^%^24direct^%^22^%^2C^%^22^%^24initial_referring_domain^%^22^%^3A^%^20^%^22^%^24direct^%^22^%^7D',
    'session': '.eJw9jzFvgzAQhf9K5ZkBDCxIHVoZoka9i2hNLd-CKCEFGxIpkBKI8t9LGKqb3nDfe9-N5Ydz1dcsOhRtXzksb_YsurGnbxYx3HwZMqUPc-aiiKfdJh5RZaGe90bzt5HMi08maWHWy2Ujytqigivw1NMmDkkCR55Y6GIfOhhJ_oRatjWpbUuKDHQpJ1Mb5FuL3ZJFGpAo3Z1Mp_WPg0cqu5JKfZwTi-KjBbHwZWK1zAItdYjitUYJz-zusKIcmt8qL8rydDkOq4nnsL7q--Z0zG01_ZvBTJ1W8YSmHJYGHz9dd2ke36X1QdgBJdlljfcwJZWYB_3SV-eVybwwYPc_n_ZnHQ.Da7GDg.SsrFjYV8R2JpijW6QVoQEiRbuJE',
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

# ---------- USER INPUTS START ---------- #
machine_id = 37
component_id = 73
location_id = 128
start_date = "07 Sep 2017 00:00:00 GMT"
end_date = "10 Oct 2017 00:00:00 GMT"
# ---------- USER INPUTS END ----------- #

# Get machine info
request_url = 'https://api.petasense.com/webapp/machines/' + str(machine_id) + '/info'
machine_info = requests.get(request_url, headers=headers, cookies=cookies)
machine_info = machine_info.json()

# Get machine configuration
params = (
    ('machine_id', str(machine_id)),
)
request_url = 'https://api.petasense.com/webapp/machine-graphics/get-config'
machine_config = requests.get(request_url, headers=headers, params=params, cookies=cookies)
machine_config = machine_config.json()

# Get measurement locations
request_url = 'https://api.petasense.com/webapp/machines/' + str(machine_id) + '/measurement-locations'
measurement_locs = requests.get(request_url, headers=headers, cookies=cookies)
measurement_locs = measurement_locs.json()

# Get broadband trend (used to extract measurement_ids)
params = (
    ('amp_type', 'velocity'),
    ('axis', 'x'),
    ('channel_type', 'low_bandwidth'),
    ('feature', 'rms'),
)
request_url = 'https://api.petasense.com/webapp/vibration-data/' + str(location_id) + '/broadband-trend'
broadband_trend = requests.get(request_url, headers=headers, params=params, cookies=cookies)
broadband_trend = broadband_trend.json()

# Find measurement ids between start and end date
dates = broadband_trend["trend_data"]["time"]
dates = [s[5:] for s in dates]
dates = [datetime.strptime(s, "%d %b %Y %H:%M:%S %Z") for s in dates]
dates = np.array(dates)

# Mask array of time of interest
start_date = datetime.strptime(start_date, "%d %b %Y %H:%M:%S %Z")
end_date = datetime.strptime(end_date, "%d %b %Y %H:%M:%S %Z")
mask = (dates > start_date) & (dates < end_date)

# Get measurement ids of interest
measurement_ids = np.array(broadband_trend["trend_data"]["measurement_id"])[mask]

axes = ["x", "y", "z"]
for i, measurement_id in enumerate(measurement_ids):
    print("Requesting measurement " + str(i+1) + " from " + str(len(measurement_ids)))

    for axis in axes:
        params = (
        ('amp_type', 'acceleration'),
        ('axis', axis),
        ('channel_type', 'low_bandwidth'),
        ('measurement_location_id', str(location_id)),
        )

        request_url = 'https://api.petasense.com/webapp/vibration-data/' + str(measurement_id) + '/waveform'

        waveform = requests.get(request_url, headers=headers, params=params, cookies=cookies)
        waveform = waveform.json()

        # Mote orientation wrt to component
        orientation = measurement_locs["measurement_locations"][0]["vibration_axis_labels"]["low_bandwidth"]

        # Write broadband trend data to file
        filename = "data/" + str(machine_id) + "/" + \
                             str(component_id) + "/" + \
                             str(location_id) + "/" + \
                             str(measurement_id) + "/" + \
                             orientation[axis] + ".json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(waveform, f)

# Extract additional information related to component
component_info = {'warning': machine_info["data"]["component_alarms"][str(component_id)]["warning"],
                  'critical': machine_info["data"]["component_alarms"][str(component_id)]["critical"],
                  'type': [machine_info["data"]["component_details"][i]["component_type"] \
                             for i in range(len(machine_info["data"]["component_details"])) \
                             if machine_info["data"]["component_details"][i]["id"] == component_id][0],
                  'subtype': [machine_info["data"]["component_details"][i]["component_subtype"] \
                             for i in range(len(machine_info["data"]["component_details"])) \
                             if machine_info["data"]["component_details"][i]["id"] == component_id][0],
                  'line_frequency': [machine_info["data"]["component_details"][i]["properties"]["line_frequency"] \
                             for i in range(len(machine_info["data"]["component_details"])) \
                             if machine_info["data"]["component_details"][i]["id"] == component_id][0],
                  'power': [machine_info["data"]["component_details"][i]["properties"]["power"] \
                             for i in range(len(machine_info["data"]["component_details"])) \
                             if machine_info["data"]["component_details"][i]["id"] == component_id][0],
                  'speed': [machine_info["data"]["component_details"][i]["properties"]["speed"] \
                             for i in range(len(machine_info["data"]["component_details"])) \
                             if machine_info["data"]["component_details"][i]["id"] == component_id][0],
                  'vfd': [machine_info["data"]["component_details"][i]["properties"]["vfd"] \
                             for i in range(len(machine_info["data"]["component_details"])) \
                             if machine_info["data"]["component_details"][i]["id"] == component_id][0],
                  'foundation': machine_info["data"]["machine_details"]["foundation"],
                  'orientation': machine_info["data"]["machine_details"]["orientation"]}

# Write component info to file
filename = "data/" + str(machine_id) + "/" + \
                     str(component_id) + "/" + \
                     "component_info.json"
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, "w") as f:
    json.dump(component_info, f)
