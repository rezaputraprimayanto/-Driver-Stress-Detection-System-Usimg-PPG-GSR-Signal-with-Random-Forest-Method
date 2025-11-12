import spidev

import csv

import datetime

import pandas as pd

import numpy as np

from scipy.signal import savgol_filter

import time



spi = spidev.SpiDev()

spi.open(0, 0)

spi.max_speed_hz = 1000000



# Membaca nilai ADC

def read_adc(channel):

    cmd = [1, (8 + channel) << 4, 0]

    adc = spi.xfer2(cmd)

    return ((adc[1] & 3) << 8) + adc[2]



# Menghitung konduktansi (dalam µS)

def calculate_conductance(adc_value):

    voltage = (adc_value * 3.3) / 1023.0

    return 1.0 / (voltage + 0.00001) if voltage > 0 else 0



# Preproses data GSR dan ekstraksi fitur

def preprocess_gsr_data(input_csv, output_csv):

    try:

        df = pd.read_csv(input_csv)

        # Memastikan data mencukupi untuk filter

        if len(df['Conductance (µS)']) < 5:

            print("Data GSR tidak mencukupi untuk filtering.")

            return None
        
        max_window_length = len(df['Conductance (µS)'])
        window_length = min(51, max_window_length)

        if window_length % 2 == 0:

            window_length -= 1

        df['Filtered_Conductance'] = savgol_filter(df['Conductance (µS)'], window_length, 2)

        features = extract_features(df)

        features.to_csv(output_csv, index=False)

        return features

    except Exception as e:

        print(f"Kesalahan saat memproses data GSR: {e}")

        return None



# Ekstraksi fitur GSR

def extract_features(df):

    features = {

        "Mean SCL": df["Filtered_Conductance"].mean(),

        "Max SCL": df["Filtered_Conductance"].max(),

        "Max SCR Peak": df["Filtered_Conductance"].idxmax(),

        "Std Dev SCR Rise Time": df["Filtered_Conductance"].std(),

        "Number of SCR Peaks": len(df[df["Filtered_Conductance"] > 0.5])

    }

    return pd.DataFrame([features])



# Fungsi untuk mengakuisisi data GSR

def acquire_gsr_data(output_csv, duration=10):

    start_time = time.time()

    with open(output_csv, mode='w', newline='') as csv_file:

        fieldnames = ['Timestamp', 'ADC Value', 'Conductance (µS)']

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()

        print(f"Memulai pengambilan data GSR ke {output_csv}...")



        while True:

            try:

                current_time = time.time()

                adc_value = read_adc(channel=0)  # GSR di channel 0

                conductance = calculate_conductance(adc_value)

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")



                # Tulis data ke file CSV

                writer.writerow({'Timestamp': timestamp, 'ADC Value': adc_value, 'Conductance (µS)': conductance})

                csv_file.flush()



                # Reset data setiap durasi tertentu

                if current_time - start_time >= duration:

                    csv_file.seek(0)

                    csv_file.truncate()

                    writer.writeheader()

                    start_time = current_time

                    

                time.sleep(0.1)  # Simulasi jeda pengambilan data

            except KeyboardInterrupt:

                print("Pengambilan data dihentikan.")

                break
