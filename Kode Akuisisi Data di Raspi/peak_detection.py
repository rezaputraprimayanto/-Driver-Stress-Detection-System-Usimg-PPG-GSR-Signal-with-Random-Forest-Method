import numpy as np
import pandas as pd
import csv
from deteksifilter import (
    butter_bandpassfilter,
    movingaverage,
    normalize_signal,
    threshold_peakdetection
)

# Fungsi untuk memuat data dari CSV yang dihasilkan program sensor
def load_data_from_csv(filename):
    """Membaca data dari file CSV yang dihasilkan program sensor"""
    timestamps = []
    heart_rates = []
    ir_data = []
    filtered_data = []
    
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            timestamps.append(row[0])
            heart_rates.append(float(row[1]) if row[1] else 0)
            ir_data.append(float(row[2]))  # Raw IR data di kolom ke-2
            filtered_data.append(float(row[3]))  # Filtered data di kolom ke-3
    
    return timestamps, heart_rates, np.array(ir_data), np.array(filtered_data)

# Fungsi untuk menjalankan deteksi puncak
def peak_detection(filename, fs=100):
    # Memuat data dari file CSV
    timestamps, heart_rates, ir_data, filtered_data = load_data_from_csv(filename)
    
    # Memilih data yang sudah difilter sebagai input untuk deteksi puncak
    preprocessed_data = filtered_data  # Atau gunakan ir_data jika perlu

    # Menjalankan metode deteksi puncak
    peaks = threshold_peakdetection(preprocessed_data, fs)
    
    return peaks

# Contoh penggunaan
filename = "sensor_data.csv"
fs = 100  # Frekuensi sampling
final_peaks = peak_detection(filename, fs)
print("Final peaks detected:", final_peaks)
