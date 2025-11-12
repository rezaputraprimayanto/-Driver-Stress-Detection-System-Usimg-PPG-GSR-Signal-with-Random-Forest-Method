import os
import csv
import numpy as np
from scipy.signal import butter, filtfilt
from main_peak_detection import ensemble_peak
from kode_kedua_main_hrv_analysis import calc_td_hrv, calc_RRI

# Fungsi untuk bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpassfilter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Fungsi untuk detrend sinyal
def detrend_signals(signals):
    x = np.arange(len(signals))
    z = np.polyfit(x, signals, 1)
    trend = z[0] * x + z[1]
    detrended = signals - trend
    return detrended

# Fungsi untuk normalisasi sinyal
def normalize_signal(data):
    data = np.array(data)
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data_normalized * 2 - 1

# Fungsi untuk menghitung fitur HRV (HR_mean, SDNN, RMSSD, pNN50)
def calc_td_hrv(RR_list_e, RR_diff, RR_sqdiff, window_size):
    """
    Menghitung empat fitur HRV utama: HR_mean, SDNN, RMSSD, pNN50.
    """
    if len(RR_list_e) == 0:
        return None
    
    HR_mean = 60000 / np.mean(RR_list_e) if np.mean(RR_list_e) > 0 else 0
    SDNN = np.std(RR_list_e)
    RMSSD = np.sqrt(np.mean(RR_sqdiff)) if len(RR_sqdiff) > 0 else 0
    pNN50 = (np.sum(np.array(RR_diff) > 50) / len(RR_diff)) * 100 if len(RR_diff) > 0 else 0

    return {
        "HR_mean": HR_mean,
        "SDNN": SDNN,
        "RMSSD": RMSSD,
        "pNN50": pNN50
    }

# Fungsi untuk menghitung heart rate (HR) dari puncak yang terdeteksi
def calculate_hr_from_peaks(peaks, fs, window_size, previous_hr=None):
    if len(peaks) < 2:
        return 0
    
    intervals = np.diff(peaks)
    intervals_sec = intervals / fs
    hrs = 60 / intervals_sec
    
    if previous_hr is not None and previous_hr > 0:
        valid_hrs = hrs[(hrs >= previous_hr*0.7) & (hrs <= previous_hr*1.3)]
        if len(valid_hrs) == 0:
            valid_hrs = hrs[(hrs >= 50) & (hrs <= 150)]
    else:
        valid_hrs = hrs[(hrs >= 50) & (hrs <= 150)]
    
    if len(valid_hrs) < 1:
        return 0
    
    return np.median(valid_hrs)

# Fungsi untuk mendeteksi keberadaan jari
def detect_finger_presence(ir_data, threshold_low=30000, threshold_high=170000, std_threshold=10000):
    """
    Deteksi keberadaan jari berdasarkan nilai IR.
    """
    mean_ir = np.mean(ir_data)
    std_ir = np.std(ir_data)
    
    if mean_ir < threshold_low or mean_ir > threshold_high:
        return False, "Silakan letakkan jari Anda pada sensor"
    
    if std_ir > std_threshold:
        return False, "Terlalu banyak gerakan, harap jangan gerakkan jari Anda"
        
    return True, "Jari terdeteksi"
