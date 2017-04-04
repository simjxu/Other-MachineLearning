import json
import numpy as np
import matplotlib.pyplot as plt

# 'r' mode for reading, 'w' mode for writing, read and write 'r+', append 'a'
# jsonfile = open('C:\Users\Simon\Desktop\Petasense\Data\measurement_data_40_L3_z.json', 'r')

# fieldnames = ("FirstName", "LastName", "IDNumber", "Message")
# reader = csv.DictReader(csvfile, fieldnames)
# out = json.dumps( [ row for row in reader ] )
# jsonfile.write(out)

# fieldnames = ("FirstName","LastName","IDNumber","Message")
# reader = csv.DictReader(csvfile, fieldnames)
# for row in reader:
#     json.dump(row, jsonfile)
#     jsonfile.write('\n')

f = open("C:\\Users\\Simon\\Documents\\Data\\measurement_data_40_L3_z.json", "r")
s = f.read()
json_dict = json.loads(s)

num_feat = 8
num_meas = 1000
trainmatx = np.matrix([[0.0 for col in range(num_feat)] for row in range(num_meas)])
for i in range(num_meas):
    trainmatx[i, 0] = json_dict['measurements'][i]['data']['z']['time_domain_features']['p2p']
    trainmatx[i, 1] = json_dict['measurements'][i]['data']['z']['time_domain_features']['rms']
    trainmatx[i, 2] = json_dict['measurements'][i]['data']['z']['time_domain_features']['peak']
    trainmatx[i, 3] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_1x']
    trainmatx[i, 4] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_2x']
    trainmatx[i, 5] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_3x']
    trainmatx[i, 6] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['output_shaft_3x']
    trainmatx[i, 7] = json_dict['measurements'][i]['data']['z']['frequency_domain_features']['shaft_3x_to_10x_sum']

t = np.arange(1, num_meas+1, 1)
plt.plot(t, trainmatx[:,1], 'ro')
plt.show()
# Only index on the measurement number (0 to 1117)
# print(json_dict['measurements'][0]['timestamp'])
# print(json_dict['measurements'][0]['data']['z']['time_domain_features']['crest_factor'])

# # Find out how many measurements
# print(len(json_dict['measurements']))

"""
Data format from Rama's JSON files
location_name
machine_name
location_id
measurements
[
    {sampling_size
    timestamp
    data
        {z
            {time_domain_features
                {crest_factor
                p2p
                peak
                rms
                }
            time_domain
                [ ...
                  ...
                ]
            frequency_domain_features
                {output_shaft_1x
                output_shaft_2x
                output_shaft_3x
                shaft_1x
                shaft_2x
                shaft_3x
                shaft_3x_to_10x_sum
                }
            frequency_domain
                {amps
                    [ ...
                      ...
                    ]
                fs
                    [ ...
                      ...
                    ]
                }
            }
        }
    }
    sampling_rate
}



"""