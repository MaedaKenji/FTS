#BACKUP CODE


# import pandas as pd
# import numpy as np
# import math

# # Membaca data dari file CSV
# file_path = 'harga_singkong.csv'
# df = pd.read_csv(file_path)

# # Menghitung jumlah interval menggunakan rumus Sturges
# n = len(df)  # jumlah total data
# k = 1 + 3.322 * math.log10(n)

# # Membulatkan jumlah interval ke bilangan bulat terdekat
# k = int(round(k))

# # Menghitung lebar interval
# range_data = df['Harga'].max() - df['Harga'].min()
# width = range_data / k

# # Menentukan batas-batas interval dan klasifikasi
# intervals = []
# start = df['Harga'].min()
# for i in range(k):
#     end = start + width
#     intervals.append((start, end))
#     start = end

# # Fungsi untuk menentukan interval
# def determine_interval(value, intervals):
#     for i, interval in enumerate(intervals):
#         if interval[0] <= value < interval[1]:
#             return f'A{i+1}'
#     return f'A{k}'  # untuk nilai yang tepat sama dengan batas atas

# # Klasifikasikan harga ke dalam interval
# df['Klasifikasi'] = df['Harga'].apply(lambda x: determine_interval(x, intervals))

# # Menghitung median interval
# interval_medians = {}
# for i, interval in enumerate(intervals):
#     median = (interval[0] + interval[1]) / 2
#     interval_medians[f'A{i+1}'] = median
#     print(f"Interval A{i+1}: {interval[0]} - {interval[1]}, Median: {median}")

# # Menentukan hubungan logika fuzzy
# relationships = []
# for i in range(len(df) - 1):
#     current_class = df.iloc[i]['Klasifikasi']
#     next_class = df.iloc[i + 1]['Klasifikasi']
#     relationships.append((current_class, next_class))

# # Menyimpan hubungan logika fuzzy ke DataFrame untuk analisis lebih lanjut
# flr_df = pd.DataFrame(relationships, columns=['Current', 'Next'])

# # Initialize a dictionary to store the relationships
# relationship_dict = {}
# for current, next_ in relationships:
#     if current not in relationship_dict:
#         relationship_dict[current] = set()
#     relationship_dict[current].add(next_)

# # Mencari nilai FLRG menggunakan mean interval
# flrg = {}
# for current in relationship_dict:
#     next_intervals = list(relationship_dict[current])
#     relevant_medians = [interval_medians[interval] for interval in next_intervals]
#     flrg[current] = np.mean(relevant_medians)

# # Fungsi untuk memprediksi harga berdasarkan input harga
# def predict_next_price(input_price, intervals, interval_medians, flrg):
#     input_interval = determine_interval(input_price, intervals)
#     if input_interval in flrg:
#         return flrg[input_interval]
#     else:
#         return None

# # Melakukan prediksi untuk setiap nilai dalam dataset
# predicted_prices = []
# for i in range(len(df) - 1):  # Menggunakan data hingga n-1 untuk memprediksi n
#     current_price = df.iloc[i]['Harga']
#     predicted_price = predict_next_price(current_price, intervals, interval_medians, flrg)
#     predicted_prices.append(predicted_price)

# # Menambahkan prediksi ke DataFrame untuk perbandingan
# df_predictions = df.iloc[1:].copy()  # Menghindari shifting satu ke depan
# df_predictions['Predicted'] = predicted_prices

# # Menghitung MAPE
# df_predictions.dropna(inplace=True)  # Menghapus baris dengan prediksi NaN
# actual_prices = df_predictions['Harga']
# predicted_prices = df_predictions['Predicted']

# mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

# print(f"MAPE: {mape:.2f}%")

# # Menampilkan hasil FLRG
# print("\nNilai FLRG:")
# for key in sorted(flrg.keys()):
#     print(f"{key} -> {flrg[key]}")



# def ensure_date_format(date_str, year):
#     try:
#         # Coba konversi ke format %d-%b-%Y
#         return pd.to_datetime(date_str, format='%d-%b-%Y')
#     except ValueError:
#         try:
#             # Jika gagal, cek apakah sudah dalam format %Y-%m-%d
#             return pd.to_datetime(date_str, format='%Y-%m-%d')
#         except ValueError:
#             try:
#                 # Jika gagal, coba format m/d/y
#                 return pd.to_datetime(date_str, format='%m/%d/%Y')
#             except ValueError:
#                 try:
#                     # Jika gagal, coba format m/d/y tanpa tahun
#                     return pd.to_datetime(date_str + f'/{year}', format='%m/%d/%Y')
#                 except ValueError:
#                     # Jika semua upaya di atas gagal, tambahkan tahun dan konversi ke %d-%b-%Y
#                     return pd.to_datetime(date_str + f'-{year}', format='%d-%b-%Y')

# def augment_data(file_path, output_path, year=2020):
#     # Membaca data dari file CSV
#     df = pd.read_csv(file_path)
    
#     # Memastikan kolom 'Date' dalam format datetime '%d-%b-%Y' atau '%Y-%m-%d'
#     df['Date'] = df['Date'].apply(lambda x: ensure_date_format(x, year))

#     augmented_rows = []

#     # Perulangan untuk setiap pasangan bulan
#     for i in range(len(df) - 1):
#         current_date = df.iloc[i]['Date']
#         next_date = df.iloc[i+1]['Date']
        
#         # Hitung jumlah hari antara dua tanggal
#         days_between = (next_date - current_date).days
        
#         if days_between > 1:
#             # Menghitung mean dari dua bulan berturut-turut
#             mean_value = df.iloc[i:i+2]['Harga'].mean()
            
#             # Membuat baris baru dengan mean tersebut
#             new_row = df.iloc[i].copy()
#             new_row['Harga'] = mean_value
#             new_row['Date'] = current_date + pd.DateOffset(days=1)
            
#             augmented_rows.append(new_row)
    
#     # Membuat DataFrame dari baris hasil augmentasi
#     df_augmented = pd.DataFrame(augmented_rows)
    
#     # Menggabungkan data asli dengan data augmentasi dan mengurutkan berdasarkan tanggal
#     df_combined = pd.concat([df, df_augmented]).sort_values(by='Date').reset_index(drop=True)
    
#     # Menyimpan data gabungan ke file CSV baru
#     df_combined.to_csv(output_path, index=False)
    
#     return df_combined


# # Menentukan path file input dan output
# input_file_path = 'harga_singkong.csv'
# output_file_path = 'harga_singkong_augmented.csv'
# input_file_path2 = 'harga_singkong_augmented.csv'
# output_file_path2 = 'harga_singkong_augmented2.csv'

# # Memanggil fungsi untuk augmentasi data
# # df_combined = augment_data(input_file_path, output_file_path)
# df_combined = augment_data(input_file_path2, output_file_path2)

# print("Data gabungan telah disimpan ke file:", output_file_path)
# print(df_combined)