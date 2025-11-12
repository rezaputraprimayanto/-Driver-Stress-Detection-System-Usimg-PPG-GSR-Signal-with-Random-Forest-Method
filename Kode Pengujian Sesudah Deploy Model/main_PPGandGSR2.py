import os
import csv
import time
import numpy as np
import threading
import pandas as pd
from queue import Queue
from sklearn.ensemble import RandomForestClassifier
import joblib
from kode_filtering_noise_PPG import (
    detrend_signals,
    normalize_signal,
    butter_bandpassfilter,
    ensemble_peak,
    calculate_hr_from_peaks,
    calc_td_hrv,
    calc_RRI,
    detect_finger_presence,
)
import main_GSR_data_acquisition  # Import untuk memanggil pembacaan GSR
import timeit
import pygame  # Untuk memutar file MP3

# Inisialisasi pygame mixer
pygame.mixer.init()

def play_audio(file_path):
    """
    Memutar file audio MP3 yang diberikan.
    """
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Tunggu hingga selesai
            time.sleep(0.1)
    except Exception as e:
        print(f"Kesalahan saat memutar audio: {e}")

# Mapping tingkat stres dari angka ke string
STRESS_LEVELS = {
    0: "Normal",
    1: "Stres Rendah",
    2: "Stres Sedang",
    3: "Stres Tinggi"
}

class MainApp:
    def __init__(self, model_path, training_csv, window_size=400, fs=100):
        self.data_queue = Queue()
        self.model_path = model_path
        self.training_csv = training_csv
        self.window_size = window_size
        self.fs = fs
        self.rf_model = None
        self.computation_times = []  # List untuk menyimpan waktu komputasi tiap iterasi klasifikasi
        self.sensor = None
        self.running = True  # Flag untuk menghentikan loop
        self.gsr_features = None  # Menyimpan fitur GSR
        self.feature_names = [
            "HR_mean", "SDNN", "RMSSD", "pNN50",
            "Mean SCL", "Max SCL", "Max SCR Peak", "Std Dev SCR Rise Time", "Number of SCR Peaks", "Stress Level"
        ]

    def initialize_sensor(self):
        try:
            import max30102
            self.sensor = max30102.MAX30102()
            print("Sensor MAX30102 berhasil diinisialisasi.")
        except ImportError:
            print("Modul MAX30102 tidak ditemukan. Pastikan sensor diatur dengan benar.")
            self.sensor = None

    def load_rf_model(self):
        try:
            self.rf_model = joblib.load(self.model_path)
            print(f"Model Random Forest berhasil dimuat dari {self.model_path}")
        except FileNotFoundError:
            print(f"Model tidak ditemukan di {self.model_path}. Pastikan model telah dilatih.")
            self.rf_model = None
        except Exception as e:
            print(f"Kesalahan dalam pemuatan model: {e}")

    def load_gsr_features(self, gsr_features_csv):
        try:
            gsr_df = pd.read_csv(gsr_features_csv)
            gsr_features = gsr_df.iloc[-1].values.tolist()
            return gsr_features
        except Exception as e:
            print(f"Kesalahan dalam memuat fitur GSR: {e}")
            return None

    def data_acquisition(self):
        ir_data_buffer = []
        try:
            print("Memulai pengambilan data dari sensor PPG...")
            while self.running:
                _, ir_data = self.sensor.read_sequential(100)
                ir_data_buffer.extend(ir_data)

                # Deteksi keberadaan jari
                finger_present, message = detect_finger_presence(ir_data)
                if not finger_present:
                    time.sleep(0.5)
                    continue

                if len(ir_data_buffer) >= self.window_size:
                    current_data = ir_data_buffer[-self.window_size:]
                    self.data_queue.put(current_data)
                    ir_data_buffer = ir_data_buffer[-self.window_size:]
                time.sleep(0.1)
        except Exception as e:
            print(f"Kesalahan dalam pengambilan data PPG: {e}")
        finally:
            if self.sensor:
                self.sensor.shutdown()
            print("\nPengambilan data PPG dihentikan.")

    def classify_data(self):
        previous_hr = None
        try:
            print("Memulai klasifikasi data...")
            while self.running:
                if not self.data_queue.empty():
                    current_data = self.data_queue.get()
                    try:
                        # Pemrosesan data PPG
                        detrended_data = detrend_signals(current_data)
                        normalized_data = normalize_signal(detrended_data)
                        filtered_data = butter_bandpassfilter(normalized_data, 0.8, 3.0, self.fs, order=2)
                        peaks = ensemble_peak(filtered_data, self.fs)
                        hr_calc = calculate_hr_from_peaks(peaks, self.fs, self.window_size, previous_hr)

                        if hr_calc > 0:
                            previous_hr = hr_calc
                            RR_list_e, RR_diff, RR_sqdiff = calc_RRI(peaks, self.fs)
                            features = calc_td_hrv(RR_list_e, RR_diff, RR_sqdiff, len(peaks))
                            
                            if features:
                                feature_vector_ppg = [
                                    features["HR_mean"],
                                    features["SDNN"],
                                    features["RMSSD"],
                                    features["pNN50"]
                                ]
                                main_GSR_data_acquisition.preprocess_gsr_data('gsr_data.csv', 'gsr_features.csv')
                                self.gsr_features = self.load_gsr_features('gsr_features.csv')

                                if self.gsr_features:
                                    feature_vector = feature_vector_ppg + self.gsr_features
                                    if np.any(np.isnan(feature_vector)):
                                        continue
                                    feature_df = pd.DataFrame([feature_vector], columns=self.feature_names[:-1])
                                    prediction = self.rf_model.predict(feature_df)
                                    stress_level = STRESS_LEVELS.get(prediction[0], "Tidak Diketahui")
                                    print(f"Prediksi Tingkat Stres: {stress_level}")

                                    # Output suara berdasarkan hasil klasifikasi
                                    if stress_level == "Stres Rendah":
                                        play_audio("stres_rendah.mp3")
                                    elif stress_level == "Stres Sedang":
                                        play_audio("stres_sedang.mp3")
                                    elif stress_level == "Stres Tinggi":
                                        play_audio("stres_tinggi.mp3")
                                
                                    self.update_training_data(feature_vector + [stress_level])
                                else:
                                    print("Fitur GSR tidak valid, tidak dapat dilakukan klasifikasi.")
                            else:
                                print("Fitur tidak valid, tidak dapat dilakukan klasifikasi.")
                        else:
                            print("Tidak dapat menghitung HR, sinyal tidak stabil.")
                    except Exception as e:
                        print(f"Kesalahan dalam pemrosesan data: {e}")
                else:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Kesalahan dalam klasifikasi data: {e}")
        finally:
            print("\nKlasifikasi data dihentikan.")

    def update_training_data(self, features):
        if not os.path.exists(self.training_csv) or os.path.getsize(self.training_csv) == 0:
            with open(self.training_csv, mode='w', newline='') as training_file:
                writer = csv.writer(training_file)
                writer.writerow(self.feature_names)  # Tulis header
        with open(self.training_csv, mode='a', newline='') as training_file:
            writer = csv.writer(training_file)
            writer.writerow(features)
        print(f"Data baru berhasil ditambahkan ke {self.training_csv}")

    def run(self):
        if not self.sensor:
            print("Sensor tidak diinisialisasi. Program dihentikan.")
            return
        if not self.rf_model:
            print("Model Random Forest tidak dimuat. Program dihentikan.")
            return

        acquisition_thread_ppg = threading.Thread(target=self.data_acquisition)
        classification_thread = threading.Thread(target=self.classify_data)

        acquisition_thread_gsr = threading.Thread(target=main_GSR_data_acquisition.acquire_gsr_data, args=("gsr_data.csv",))

        acquisition_thread_ppg.start()
        acquisition_thread_gsr.start()
        classification_thread.start()

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Menghentikan program...")
            self.running = False
            acquisition_thread_ppg.join()
            acquisition_thread_gsr.join()
            classification_thread.join()
            print("Program dihentikan.")

if __name__ == "__main__":
    model_path = "rf_model.pkl"
    training_csv = "train_labeled_PPGandGSR.csv"
    app = MainApp(model_path, training_csv)
    app.initialize_sensor()
    app.load_rf_model()
    app.run()
