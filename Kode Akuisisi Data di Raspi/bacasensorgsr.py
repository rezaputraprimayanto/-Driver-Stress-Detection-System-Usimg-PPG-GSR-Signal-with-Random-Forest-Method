import spidev
import time
import csv
import datetime

# Inisialisasi SPI
spi = spidev.SpiDev()
spi.open(0, 0)  # Bus 0, Device 0
spi.max_speed_hz = 1000000  # 1MHz

def read_adc(channel):
    """
    Membaca nilai dari MCP3008 ADC
    """
    # Command untuk membaca dari MCP3008
    cmd = [1, (8 + channel) << 4, 0]
    adc = spi.xfer2(cmd)
    
    # Mendapatkan 10-bit data
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

def calculate_conductance(adc_value):
    """
    Mengkonversi nilai ADC ke conductance (mikroSiemens)
    """
    # Voltage = (ADC Value * 3.3) / 1023
    voltage = (adc_value * 3.3) / 1023.0
    
    # Conductance = 1/(voltage*1M) = 1/voltage mikroSiemens
    # Menambahkan offset kecil untuk menghindari pembagian dengan nol
    conductance = 1.0 / (voltage + 0.00001) if voltage > 0 else 0
    return conductance

def print_terminal_output(timestamp, adc_value, conductance):
    """
    Fungsi untuk mencetak output ke terminal dengan format yang rapi
    """
    print(f"Timestamp: {timestamp}")
    print(f"ADC Value: {adc_value}")
    print(f"Conductance: {conductance:.2f} µS")
    print("-" * 50)

def main():
    # Membuat file CSV dan menulis header
    with open('gsr_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'ADC Value', 'Conductance (µS)'])
    
    print("Mulai pembacaan data. Tekan Ctrl+C untuk menghentikan dan menyimpan data ke CSV.")
    
    # Loop utama untuk membaca data sensor
    try:
        while True:
            # Membaca data dari sensor
            adc_value = read_adc(0)  # Channel 0
            conductance = calculate_conductance(adc_value)
            
            # Mendapatkan timestamp
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            
            # Cetak output ke terminal
            print_terminal_output(timestamp, adc_value, conductance)
            
            # Delay untuk pembacaan berikutnya (sesuaikan sesuai kebutuhan)
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nProgram dihentikan. Menyimpan data ke CSV...")
        with open('gsr_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, adc_value, conductance])
        print("Data disimpan dalam 'gsr_data.csv'")
        spi.close()

if __name__ == "__main__":
    main()
