import json
import os

import helpers
import numpy as np
import pandas as pd

base_dir = 'data/'

machines = os.listdir(base_dir)
for machine in machines:
    machine_dir = os.path.join(base_dir, machine)

    components = os.listdir(machine_dir)
    for component in components:
        component_dir = os.path.join(machine_dir, component)
        component_info = json.load(open(os.path.join(component_dir, 'component_info.json')))

        locations = os.listdir(component_dir)
        locations.remove('component_info.json')
        for location in locations:
            location_dir = os.path.join(component_dir, location)

            measurements = os.listdir(location_dir)
            measurements.sort()

            # Constuct a dataframe containing the features of all measurements
            # of the given location
            location_feature_dataset = pd.DataFrame()
            for measurement in measurements:
                measurement_dir = os.path.join(location_dir, measurement)

                # The axes of the current measurement
                directions = os.listdir(measurement_dir)

                feature_names = []
                features = []
                for direction in directions:
                    direction_data_f = os.path.join(measurement_dir, direction)
                    print(direction_data_f)
                    direction_data = json.load(open(direction_data_f))

                    direction_feature_names, direction_features = helpers.get_feature_vector(
                        direction_data['waveform_data'],
                        direction_data['sampling_rate'],
                        direction[:-5])

                    # Extend current axis features to the measurement features
                    feature_names.extend(direction_feature_names)
                    features.extend(direction_features)

                # Construct a series object containing the features of all
                # axes and append it to the location data frame
                feature_series = pd.Series(features, feature_names)
                location_feature_dataset = location_feature_dataset.append(feature_series, ignore_index=True)

            # Compute the average ISO levels used for feature scaling
            rms_mean = np.mean(np.array([
                np.mean(location_feature_dataset['Axial_velocity_rms']),
                np.mean(location_feature_dataset['Radial_velocity_rms']),
                np.mean(location_feature_dataset['Tangential_velocity_rms'])]))
            warning_level = np.mean(np.array([
                component_info['warning']['Axial'],
                component_info['warning']['Radial'],
                component_info['warning']['Tangential']]))
            critical_level = np.mean(np.array([
                component_info['critical']['Axial'],
                component_info['critical']['Radial'],
                component_info['critical']['Tangential']]))
            worst_level = 1.25 * critical_level

            # Compute scaled features
            feature_mean = np.mean(location_feature_dataset, axis=0)
            warning_location_feature_dataset = (location_feature_dataset - feature_mean) + feature_mean * warning_level/rms_mean
            critical_location_feature_dataset = (location_feature_dataset - feature_mean) + feature_mean * critical_level/rms_mean
            worst_location_feature_dataset = (location_feature_dataset - feature_mean) + feature_mean * worst_level/rms_mean

            # Save features to file
            filename = 'features/' + location + '/baseline.csv'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            location_feature_dataset.to_csv(filename, index=False)

            filename = 'features/' + location + '/warning.csv'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            warning_location_feature_dataset.to_csv(filename, index=False)

            filename = 'features/' + location + '/critical.csv'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            critical_location_feature_dataset.to_csv(filename, index=False)

            filename = 'features/' + location + '/worst.csv'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            worst_location_feature_dataset.to_csv(filename, index=False)
