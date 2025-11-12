import math
import numpy as np
import pandas as pd
import nolds

from scipy.interpolate import UnivariateSpline
from scipy import stats

def calc_RRI(peaklist, fs):
    RR_list = []
    RR_list_e = []
    cnt = 0
    while (cnt < (len(peaklist)-1)):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt]) #Calculate distance between beats in # of samples
        ms_dist = ((RR_interval / fs) * 1000.0)  #fs? ??? 1???? ???? -> 1ms??? change /  Convert sample distances to ms distances
        cnt += 1
        RR_list.append(ms_dist)
    mean_RR = np.mean(RR_list)

    for ind, rr in enumerate(RR_list):
        if rr >  mean_RR - 300 and rr < mean_RR + 300:
            RR_list_e.append(rr)
            
    RR_diff = []
    RR_sqdiff = []
    cnt = 0
    while (cnt < (len(RR_list_e)-1)):
        RR_diff.append(abs(RR_list_e[cnt] - RR_list_e[cnt+1]))
        RR_sqdiff.append(math.pow(RR_list_e[cnt] - RR_list_e[cnt+1], 2))
        cnt += 1
        
    return RR_list_e, RR_diff, RR_sqdiff

def calc_heartrate(RR_list):
    HR = []
    heartrate_array=[]
    window_size = 10

    for val in RR_list:
        if val > 400 and val < 1500:
            heart_rate = 60000.0 / val #60000 ms (1 minute) / ?? beat??? ??? ??
        # if RR-interval < .1905 seconds, heart-rate > highest recorded value, 315 BPM. Probably an error!
        elif (val > 0 and val < 400) or val > 1500:
            if len(HR) > 0:
                # ... and use the mean heart-rate from the data so far:
                heart_rate = np.mean(HR[-window_size:])

            else:
                heart_rate = 60.0
        else:
            # Get around divide by 0 error
            print("err")
            heart_rate = 0.0

        HR.append(heart_rate)

    return HR

    
def calc_td_hrv(RR_list, RR_diff, RR_sqdiff, window_length): 
    
    # Time
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
    
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    

    features = {'HR_mean': HR_mean, 'HR_std': HR_std, 'meanNN': meanNN, 'SDNN': SDNN, 'medianNN': medianNN,
                'meanSD': meanSD, 'SDSD': SDSD, 'RMSSD': RMSSD, 'pNN20': pNN20, 'pNN50': pNN50, 'TINN': TINN}

    return features


# def calc_fd_hrv(RR_list):  
    # rr_x = []
    # pointer = 0
    # for x in RR_list:
        # pointer += x
        # rr_x.append(pointer)
        
    # if len(rr_x) <= 3 or len(RR_list) <= 3:
        # print("rr_x or RR_list less than 5")   
        # return 0
    
    # RR_x_new = np.linspace(rr_x[0], rr_x[-1], int(rr_x[-1]))
    
    # try:
        # interpolated_func = UnivariateSpline(rr_x, RR_list, k=3, s=len(rr_x))
    # except Exception as e:
        # print(f"Interpolasi spline gagal: {e}")
        # return 0
    
    # Y = np.fft.fft(interpolated_func(RR_x_new)) / len(RR_x_new)
    # psd = np.power(Y[range(len(Y) // 2)], 2)
    # frq = np.fft.fftfreq(len(RR_x_new), d=((1 / 1000.0)))
    # frq = frq[range(len(frq) // 2)]

    # # Periksa apakah PSD atau frekuensi kosong atau tidak valid
    # if np.sum(psd) == 0 or len(psd) == 0 or len(frq) == 0:
        # print("Tidak ada spektrum yang valid untuk analisis.")
        # return 0

    # lf = np.trapz(abs(psd[(frq >= 0.04) & (frq <= 0.15)]))
    # hf = np.trapz(abs(psd[(frq > 0.15) & (frq <= 0.5)]))
    # ulf = np.trapz(abs(psd[frq < 0.003]))
    # vlf = np.trapz(abs(psd[(frq >= 0.003) & (frq < 0.04)]))

    # if hf != 0:
        # lfhf = lf / hf
    # else:
        # lfhf = 0

    # total_power = lf + hf + vlf
    # if total_power > 0:
        # lfp = lf / total_power
        # hfp = hf / total_power
    # else:
        # lfp = 0
        # hfp = 0

    # features = {'LF': lf, 'HF': hf, 'ULF': ulf, 'VLF': vlf, 'LFHF': lfhf, 'total_power': total_power, 'lfp': lfp, 'hfp': hfp}

    # # Gantikan NaN dengan nol
    # for key in features:
        # if np.isnan(features[key]):
            # features[key] = 0

    # return features

