# -*- coding: utf-8 -*-

import numpy as np
import csv
import math
from main_peak_detection import load_data_from_csv, analyze_peaks, calculate_hrv_metrics
from deteksifilter import movingaverage, threshold_peakdetection
from scipy.interpolate import UnivariateSpline
import nolds

def calc_RRI(peaklist, fs):
    """
    Menghitung RR-intervals dari daftar puncak.
    """
    RR_list = []
    RR_list_e = []
    cnt = 0
    while cnt < (len(peaklist) - 1):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt])  # Jarak antar puncak dalam jumlah sampel
        ms_dist = ((RR_interval / fs) * 1000.0)  # Konversi jarak sampel ke jarak dalam milidetik
        cnt += 1
        RR_list.append(ms_dist)
    mean_RR = np.mean(RR_list)

    for ind, rr in enumerate(RR_list):
        if mean_RR - 300 < rr < mean_RR + 300:
            RR_list_e.append(rr)
            
    RR_diff = [abs(RR_list_e[i] - RR_list_e[i+1]) for i in range(len(RR_list_e) - 1)]
    RR_sqdiff = [math.pow(RR_list_e[i] - RR_list_e[i+1], 2) for i in range(len(RR_list_e) - 1)]
        
    return RR_list_e, RR_diff, RR_sqdiff

def calc_heartrate(RR_list):
    """
    Menghitung heart rate dari daftar RR-intervals.
    """
    HR = []
    window_size = 10

    for val in RR_list:
        if 400 < val < 1500:
            heart_rate = 60000.0 / val  # 60000 ms (1 menit) / jarak antar puncak dalam milidetik
        elif 0 < val < 400 or val > 1500:
            if len(HR) > 0:
                heart_rate = np.mean(HR[-window_size:])
            else:
                heart_rate = 60.0
        else:
            print("err")
            heart_rate = 0.0

        HR.append(heart_rate)

    return HR

def approximate_entropy(U, m=2, r=3):
    """
    Menghitung approximate entropy dari deret RR-intervals.
    """
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m+1) - _phi(m))

def shannon_entropy(signal):
    """
    Menghitung shannon entropy dari deret RR-intervals.
    """
    data_set = list(set(signal))
    freq_list = [float(signal.count(entry)) / len(signal) for entry in data_set]
    ent = -sum(freq * np.log2(freq) for freq in freq_list)
    return ent

def sample_entropy(sig, ordr, tor):
    """
    Menghitung sample entropy dari deret RR-intervals.
    """
    sig = np.array(sig)
    n = len(sig)
    matchnum = 0.0
    for i in range(n-ordr):
        tmpl = sig[i:i+ordr]
        for j in range(i+1, n-ordr+1): 
            ltmp = sig[j:j+ordr]
            diff = tmpl - ltmp
            if all(diff < tor):
                matchnum += 1
    allnum = (n - ordr + 1) * (n - ordr) / 2
    return -math.log(matchnum / allnum) if matchnum >= 0.1 else 1000.0

def calc_td_hrv(RR_list, RR_diff, RR_sqdiff, window_length):
    """
    Menghitung time-domain HRV features.
    """
    HR = calc_heartrate(RR_list)
    HR_mean, HR_std = np.mean(HR), np.std(HR)
    meanNN, SDNN, medianNN = np.mean(RR_list), np.std(RR_list), np.median(np.abs(RR_list))
    meanSD, SDSD = np.mean(RR_diff) , np.std(RR_diff)
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    
    NN20 = [x for x in RR_diff if x > 20]
    NN50 = [x for x in RR_diff if x > 50]
    pNN20 = len(NN20) / window_length
    pNN50 = len(NN50) / window_length
    
    bar_y, bar_x = np.histogram(RR_list)
    TINN = np.max(bar_x) - np.min(bar_x)

    return {'HR_mean': HR_mean, 'HR_std': HR_std, 'meanNN': meanNN, 'SDNN': SDNN, 'medianNN': medianNN,
            'meanSD': meanSD, 'SDSD': SDSD, 'RMSSD': RMSSD, 'pNN20': pNN20, 'pNN50': pNN50, 'TINN': TINN}

def calc_fd_hrv(RR_list):
    """
    Menghitung frekuensi-domain HRV features.
    """
    rr_x = [sum(RR_list[:i+1]) for i in range(len(RR_list))]
        
    if len(rr_x) <= 3 or len(RR_list) <= 3:
        print("rr_x or RR_list less than 5")   
        return 0
    
    RR_x_new = np.linspace(rr_x[0], rr_x[-1], int(rr_x[-1]))
    interpolated_func = UnivariateSpline(rr_x, RR_list, k=3)
    
    datalen = len(RR_x_new)
    frq = np.fft.fftfreq(datalen, d=((1/1000.0)))
    frq = frq[range(int(datalen/2))]
    Y = np.fft.fft(interpolated_func(RR_x_new))/datalen
    Y = Y[range(int(datalen/2))]
    psd = np.power(Y, 2)

    lf = np.trapz(abs(psd[(frq >= 0.04) & (frq <= 0.15)]))
    hf = np.trapz(abs(psd[(frq > 0.15) & (frq <= 0.5)]))
    ulf = np.trapz(abs(psd[frq < 0.003]))
    vlf = np.trapz(abs(psd[(frq >= 0.003) & (frq < 0.04)]))
    
    lfhf = lf / hf if hf != 0 else 0
    total_power = lf + hf + vlf
    lfp = lf / total_power
    hfp = hf / total_power

    return {'LF': lf, 'HF': hf, 'ULF' : ulf, 'VLF': vlf, 'LFHF': lfhf, 'total_power': total_power, 'lfp': lfp, 'hfp': hfp}

def calc_nonli_hrv(RR_list, label):
    """
    Menghitung non-linear HRV features.
    """
    diff_RR = np.diff(RR_list)
    sd_heart_period = np.std(diff_RR, ddof=1) ** 2
    SD1 = np.sqrt(sd_heart_period * 0.5)
    SD2 = 2 * sd_heart_period - 0.5 * sd_heart_period
    pA = SD1 * SD2
    pQ = SD1 / SD2 if SD2 != 0 else 0
    ApEn = approximate_entropy(RR_list, 2, 3)  
    shanEn = shannon_entropy(RR_list)
    D2 = nolds.corr_dim(RR_list, emb_dim=2)

    return {'SD1': SD1, 'SD2': SD2, 'pA': pA, 'pQ': pQ, 'ApEn' : ApEn, 'shanEn': shanEn, 'D2': D2, 'label': label}

def get_window_stats_27_features(ppg_seg, window_length, label, ensemble, ma_usage):
    """
    Ekstraksi fitur HRV dari segmen data PPG.
    """
    fs = 64  
    
    if ma_usage:
        fwd = movingaverage(ppg_seg, periods=3)
        bwd = movingaverage(ppg_seg[::-1], periods=3)
        ppg_seg = np.mean(np.vstack((fwd, bwd[::-1])), axis=0)
    ppg_seg = np.array([item.real for item in ppg_seg])
    
    peaks = threshold_peakdetection(ppg_seg, fs)

    if ensemble:
        ensemble_ths = 3
        peaks = threshold_peakdetection(ppg_seg, fs, ensemble_ths)
        
        if len(peaks) < 100:
            print("skip")
            return []

    RR_list, RR_diff, RR_sqdiff = calc_RRI(peaks, fs)
    
    if len(RR_list) <= 3:
        return {}
    
    td_features = calc_td_hrv(RR_list, RR_diff, RR_sqdiff, window_length)
    fd_features = calc_fd_hrv(RR_list)
    
    if fd_features == 0:
        return {}
    
    nonli_features = calc_nonli_hrv(RR_list, label)
    
    return {**td_features, **fd_features, **nonli_features}

def save_features_to_csv(features, output_filename):
    """
    Menyimpan fitur HRV ke file CSV.
    """
    fieldnames = list(features[0].keys())
    fieldnames.append('timestamp')
    
    with open(output_filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for feature, timestamp in zip(features, timestamps):
            feature['timestamp'] = timestamp
            writer.writerow(feature)

def main():
    try:
        filename = "sensor_data.csv"
        timestamps, heart_rates, ir_data, filtered_data = load_data_from_csv(filename)
        
        window_length = 1
        ensemble = True
        ma_usage = True
        
        features = []
        
        for i in range(0, len(ir_data), window_length * 64):
            window_data = ir_data[i:i + window_length * 64]
            window_timestamps = timestamps[i:i + window_length * 64]
            
            if len(window_data) < window_length * 64:
                break
            
            window_features = get_window_stats_27_features(window_data, window_length, -1, ensemble, ma_usage)
            
            if window_features:
                features.append(window_features)
        
        if features:
            save_features_to_csv(features, "hrv_features.csv")
            print("Fitur HRV telah disimpan ke 'hrv_features.csv'")
        else:
            print("Tidak ada fitur HRV yang dihitung")
        
    except FileNotFoundError:
        print(f"Error: File {filename} tidak ditemukan")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
