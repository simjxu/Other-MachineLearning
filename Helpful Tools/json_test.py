import json
import csv

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

f = open("measurement_data_40_L3_z.json", "r")
s = f.read()
json_dict = json.loads(s)

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