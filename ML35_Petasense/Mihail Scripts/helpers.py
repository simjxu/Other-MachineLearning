# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd

from scipy import integrate
from dsp3 import thinkdsp

# Constants
VELOCITY_SPECTRUM_BAND_SIZE = 100  # Hz
VELOCITY_SPECTRUM_BAND_STOP = 500  # Hz
ACCELERATION_SPECTRUM_BAND_SIZE = 500  # Hz
G2_CONSTANT = 1000*9.81 # mm/s^2

def get_velocity_using_integration(
    acceleration_waveform,
    sampling_rate):

    """ Compute the velocity waveform by integration of acceleration waveform.
    Inputs:
        acceleration_waveform: acceleration waveform data as numpy array
        sampling_rate: sampling rate
    Outputs:
        velocity_wave: a thinkdsp.Wave object of the velocity waveform
    """

    velocity_data = integrate.cumtrapz(
        acceleration_waveform,
        initial=acceleration_waveform[0]
    )
    velocity_data /= sampling_rate
    velocity_data = velocity_data*G2_CONSTANT

    velocity_wave = thinkdsp.Wave(
        velocity_data,
        framerate=sampling_rate
    )
    velocity_wave.detrend()

    return velocity_wave

def get_velocity_using_omega_arithmetic(
    acceleration_waveform,
    sampling_rate):

    """ Compute the velocity waveform from acceleration waveform using the
    omega arithmetic method.
    Inputs:
        acceleration_waveform: acceleration waveform data as numpy array
        sampling_rate: sampling rate
    Outputs:
        velocity_wave: a thinkdsp.Wave object of the velocity waveform
    """

    velocity_data = []
    acceleration_wave = thinkdsp.Wave(
        acceleration_waveform,
        framerate=sampling_rate
    )
    acceleration_spectrum = acceleration_wave.make_spectrum()
    for i in range(len(acceleration_spectrum.hs)):
        if acceleration_spectrum.hs[i] and acceleration_spectrum.fs[i]:
            velocity_data.append(
                acceleration_spectrum.hs[i] / (2 * math.pi * acceleration_spectrum.fs[i] * 1j)
            )
        else:
            velocity_data.append(acceleration_spectrum.hs[i])

    velocity_data = velocity_data*G2_CONSTANT
    velocity_spectrum = thinkdsp.Spectrum(
        velocity_data,
        sampling_rate
    )
    velocity_waveform = velocity_spectrum.make_wave()
    velocity_waveform.detrend()

    return velocity_waveform

def get_feature_vector(
    acceleration_waveform,
    sampling_rate,
    axis,
    method='integration'):
    """ Compute the feature vector of a given waveform.
    Inputs:
        acceleration_waveform: the vibration waveform data as numpy array
        sampling_rate: the sampling rate
        axis: the axis of the measurement (Axial, Radial, Tangential)
        method: the method used to integrate the acceleration waveform (integration, omega)
    Outputs:
        feature_vector: the feature vector of the provided waveform
    """

    acceleration_wave = thinkdsp.Wave(
        acceleration_waveform,
        framerate=sampling_rate
    )
    acceleration_spectrum = acceleration_wave.make_spectrum()

    if method == 'integration':
        velocity_wave = get_velocity_using_integration(
            acceleration_waveform,
            sampling_rate
        )
    elif method == 'omega':
        velocity_wave = get_velocity_using_omega_arithmetic(
            acceleration_waveform,
            sampling_rate
        )
    velocity_spectrum = velocity_wave.make_spectrum()

    acceleration_band_range = [VELOCITY_SPECTRUM_BAND_STOP, int(acceleration_spectrum.max_freq)]
    acceleration_band_size = ACCELERATION_SPECTRUM_BAND_SIZE
    acceleration_bands = np.arange(
        acceleration_band_range[0],
        acceleration_band_range[1],
        acceleration_band_size) + acceleration_band_size

    velocity_band_range = [0, VELOCITY_SPECTRUM_BAND_STOP]
    velocity_band_size = VELOCITY_SPECTRUM_BAND_SIZE
    velocity_bands = np.arange(
        velocity_band_range[0],
        velocity_band_range[1],
        velocity_band_size) + velocity_band_size

    feature_names = []
    feature_names.append(axis + '_acceleration_rms')
    for band in acceleration_bands:
        feature_names.append(axis + '_acceleration_' + str(band))

    feature_names.append(axis + '_velocity_rms')
    for band in velocity_bands:
        feature_names.append(axis + '_velocity_' + str(band))

    features = []
    features.append(acceleration_wave.rms())
    features.extend(acceleration_spectrum.get_band_amplitude_sum(
        acceleration_band_range[0],
        acceleration_band_range[1],
        acceleration_band_size
    ))
    features.append(velocity_wave.rms())
    features.extend(velocity_spectrum.get_band_amplitude_sum(
        velocity_band_range[0],
        velocity_band_range[1],
        velocity_band_size
    ))

    return feature_names, features
