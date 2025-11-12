#deteksifilter.py
import os
import csv
import time
import numpy as np
import max30102
import hrcalc
from scipy.stats import kurtosis, skew
from scipy.signal import butter, lfilter, filtfilt
from scipy import stats
from main_peak_detection import ensemble_peak
from main_hrv_analysis import *

# Fungsi-fungsi processing
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

def movingaverage(data, periods=4):
    result = []
    data_set = np.asarray(data)
    weights = np.ones(periods) / periods
    result = np.convolve(data_set, weights, mode='valid')
    return result

def detrend_signals(signals):
    x = np.arange(len(signals))
    z = np.polyfit(x, signals, 1)
    trend = z[0] * x + z[1]
    detrended = signals - trend
    return detrended

def normalize_signal(data):
    data = np.array(data)
    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data_normalized * 2 - 1

def threshold_peakdetection(dataset, fs):
    # Menggunakan median sebagai threshold untuk lebih robust terhadap noise
    threshold = np.median(dataset) + 0.2 * np.std(dataset)
    peaklist = []
    listpos = 0
    window = []
    
    for datapoint in dataset:
        if (datapoint < threshold) and (len(window) < 1):
            listpos += 1
        elif (datapoint >= threshold):
            window.append(datapoint)
            listpos += 1
        else:
            if window:
                maximum = max(window)
                beatposition = listpos - len(window) + (window.index(max(window)))
                peaklist.append(beatposition)
            window = []
            listpos += 1
    
    # Filter peaks yang terlalu dekat
    filtered_peaks = []
    min_distance = int(0.4 * fs)  # Meningkatkan minimal distance ke 400ms
    
    for i in range(len(peaklist)):
        if i == 0:
            filtered_peaks.append(peaklist[i])
        else:
            if peaklist[i] - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peaklist[i])
    
    return filtered_peaks

def calculate_hr_from_peaks(peaks, fs, window_size, previous_hr=None):
    if len(peaks) < 2:
        return 0
    
    # Hitung intervals antara peaks
    intervals = np.diff(peaks)
    intervals_sec = intervals / fs
    hrs = 60 / intervals_sec
    
    # Filter HR yang tidak masuk akal dengan range yang lebih ketat
    if previous_hr is not None and previous_hr > 0:
        # Jika ada previous HR, gunakan sebagai reference
        valid_hrs = hrs[(hrs >= previous_hr*0.7) & (hrs <= previous_hr*1.3)]
        if len(valid_hrs) == 0:
            # Jika tidak ada HR yang valid dalam range, gunakan range default
            valid_hrs = hrs[(hrs >= 50) & (hrs <= 150)]
    else:
        # Range default jika tidak ada previous HR
        valid_hrs = hrs[(hrs >= 50) & (hrs <= 150)]
    
    if len(valid_hrs) < 1:
        return 0
    
    # Gunakan median untuk mengurangi efek outlier
    return np.median(valid_hrs)
    
def detect_finger_presence(ir_data, threshold_low=30000, threshold_high=170000, std_threshold=10000):
    """
    Deteksi keberadaan jari berdasarkan:
    1. Rentang nilai IR dalam range yang sesuai
    2. Standar deviasi sinyal untuk deteksi gerakan
    """
    mean_ir = np.mean(ir_data)
    std_ir = np.std(ir_data)
    
    # Cek apakah nilai IR dalam range yang sesuai
    if mean_ir < threshold_low or mean_ir > threshold_high:
        return False, "Silakan letakkan jari Anda pada sensor"
    
    # Cek apakah sinyal cukup stabil
    if std_ir > std_threshold:
        return False, "Terlalu banyak gerakan, harap jangan gerakkan jari Anda"
        
    return True, "Jari terdeteksi"

# Inisialisasi sensor MAX30102
sensor = max30102.MAX30102()

# Parameter
ir_data_buffer = []
hr_data = []
fs = 100  # Sampling frequency
no_finger_counter = 0
WINDOW_SIZE = 400  # Meningkatkan window size ke 4 detik
SMOOTH_SIZE = 5  # Window size untuk smoothing HR
MAX_NO_FINGER_COUNT = 5

# Buffer untuk smoothing HR
hr_buffer = []
previous_hr = None

def smooth_hr(hr_buffer, new_hr):
    """Smoothing HR menggunakan median filter"""
    if new_hr == 0:
        return 0
        
    hr_buffer.append(new_hr)
    if len(hr_buffer) > SMOOTH_SIZE:
        hr_buffer.pop(0)
        
    return np.median(hr_buffer)

def save_to_csv(filename, timestamp, hr, ir_data, filtered_data):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, hr] + list(ir_data) + list(filtered_data))

def dict_to_csv(csv_filename, dictionary_data):
    fieldnames = dictionary_data.keys()
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(dictionary_data)

# Buat file CSV
csv_filename = "sensor_data.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Heart Rate (BPM)", "Raw IR Data", "Filtered Data"])

print("Program mulai berjalan...")
print("Tekan Ctrl+C untuk menghentikan program")

try:
    while True:
        # Baca data dari sensor
        red_data, ir_data = sensor.read_sequential(100)
        
        # Cek keberadaan jari
        finger_present, message = detect_finger_presence(ir_data)
        
        if not finger_present:
            print(message)
            no_finger_counter += 1
            if no_finger_counter >= MAX_NO_FINGER_COUNT:
                # Reset semua buffer jika tidak ada jari terdeteksi untuk beberapa waktu
                ir_data_buffer = []
                hr_buffer = []
                previous_hr = None
                no_finger_counter = 0
            time.sleep(0.1)
            continue
        else:
            no_finger_counter = 0
        
        # Tambahkan data ke buffer
        ir_data_buffer.extend(ir_data)
        
        if len(ir_data_buffer) >= WINDOW_SIZE:
            current_data = ir_data_buffer[-WINDOW_SIZE:]
            
            # Preprocessing dengan multiple filtering stages
            try:
                detrended_data = detrend_signals(current_data)
                normalized_data = normalize_signal(detrended_data)
                
                # Multiple stage filtering
                filtered_data = butter_bandpassfilter(normalized_data, 0.8, 3.0, fs, order=2)
                filtered_data = movingaverage(filtered_data, periods=3)
                
                # Deteksi peaks
                peaks = ensemble_peak(filtered_data, fs)
                
                # Hitung heart rate dengan previous HR sebagai reference
                hr_calc = calculate_hr_from_peaks(peaks, fs, WINDOW_SIZE, previous_hr)
                
                # Smooth HR
                smoothed_hr = smooth_hr(hr_buffer, hr_calc)
                
                if smoothed_hr > 0:
                    print(f"Heart Rate: {smoothed_hr:.1f} BPM")
                    previous_hr = smoothed_hr
                    hr_data.append(smoothed_hr)
                    
                    # Simpan data ke CSV
                    save_to_csv(csv_filename, 
                               time.strftime("%Y-%m-%d %H:%M:%S"), 
                               smoothed_hr, 
                               current_data, 
                               filtered_data)
                               
                    RR_list_e, RR_diff, RR_sqdiff = calc_RRI(peaks, fs)
                    td_dict = calc_td_hrv(RR_list_e, RR_diff, RR_sqdiff, WINDOW_SIZE)
                    # fd_dict = calc_fd_hrv(RR_list_e)
                    if td_dict : 
                        # total_features = {**td_dict, **fd_dict}
                        total_features = {**td_dict}
                        dict_to_csv('features.csv', total_features)
                        print("Data berhasil tersimpan")
                    else :
                        print("Data tidak tersimpan")
        
                else:
                    print("Mendeteksi heart rate...")
                
            except Exception as e:
                print(f"Error dalam pemrosesan: {e}")
                ir_data_buffer = ir_data_buffer[-WINDOW_SIZE:]  # Keep only recent data
                continue
            
            # Maintain buffer size
            ir_data_buffer = ir_data_buffer[-WINDOW_SIZE:]
        
        time.sleep(0.01)

except KeyboardInterrupt:
    sensor.shutdown()
    print("\nProgram dihentikan.")
    print(f"Data telah disimpan di {csv_filename}")
