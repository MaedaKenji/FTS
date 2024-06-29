import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

def ensure_date_format(date_str, year):
    try:
        return pd.to_datetime(date_str, format='%d-%b-%Y')
    except ValueError:
        try:
            return pd.to_datetime(date_str, format='%Y-%m-%d')
        except ValueError:
            try:
                return pd.to_datetime(date_str, format='%m/%d/%Y')
            except ValueError:
                try:
                    return pd.to_datetime(date_str + f'/{year}', format='%m/%d/%Y')
                except ValueError:
                    return pd.to_datetime(date_str + f'-{year}', format='%d-%b-%Y')

def augment_data(file_path, output_path, year=2020):
    df = pd.read_csv(file_path)
    df['Date'] = df['Date'].apply(lambda x: ensure_date_format(x, year))
    augmented_rows = []
    for i in range(len(df) - 1):
        current_date = df.iloc[i]['Date']
        next_date = df.iloc[i+1]['Date']
        days_between = (next_date - current_date).days
        if days_between > 1:
            mean_value = df.iloc[i:i+2]['Harga'].mean()
            new_row = df.iloc[i].copy()
            new_row['Harga'] = mean_value
            new_row['Date'] = current_date + pd.DateOffset(days=1)
            augmented_rows.append(new_row)
    df_augmented = pd.DataFrame(augmented_rows)
    df_combined = pd.concat([df, df_augmented]).sort_values(by='Date').reset_index(drop=True)
    df_combined.to_csv(output_path, index=False)
    return df_combined

def determine_interval(value, intervals, k):
    for i, interval in enumerate(intervals):
        if interval[0] <= value < interval[1]:
            return f'A{i+1}'
    return f'A{k}'  # untuk nilai yang tepat sama dengan batas atas

def predict_next_price(input_price, intervals, interval_medians, flrg, k):
    input_interval = determine_interval(input_price, intervals, k)
    if input_interval in flrg:
        return flrg[input_interval]
    else:
        return None

def calculate_flrg(df):
    n = len(df)
    k = 1 + 3.322 * math.log10(n)
    k = int(round(k))
    range_data = df['Harga'].max() - df['Harga'].min()
    width = range_data / k
    intervals = []
    start = df['Harga'].min()
    for i in range(k):
        end = start + width
        intervals.append((start, end))
        start = end
    interval_medians = {}
    for i, interval in enumerate(intervals):
        median = (interval[0] + interval[1]) / 2
        interval_medians[f'A{i+1}'] = median
        # print(f"A{i+1}: {interval[0]} - {interval[1]} -> {median}")
    relationships = []
    for i in range(len(df) - 1):
        current_class = determine_interval(df.iloc[i]['Harga'], intervals, k)
        next_class = determine_interval(df.iloc[i + 1]['Harga'], intervals, k)
        relationships.append((current_class, next_class))
    relationship_dict = {}
    for current, next_ in relationships:
        if current not in relationship_dict:
            relationship_dict[current] = set()
        relationship_dict[current].add(next_)
    flrg = {}
    for current in relationship_dict:
        next_intervals = list(relationship_dict[current])
        relevant_medians = [interval_medians[interval] for interval in next_intervals]
        flrg[current] = np.mean(relevant_medians)
    return flrg, intervals, interval_medians, k


def main():
    input_file_path = 'harga_singkong.csv'
    output_file_path = 'harga_singkong_augmented.csv'
    input_file_path2 = 'harga_singkong_augmented.csv'
    output_file_path2 = 'harga_singkong_augmented2.csv'
    validation_file_path = 'validation.csv'

    df_combined = augment_data(input_file_path2, output_file_path2)
    # print("Data gabungan telah disimpan ke file:", output_file_path2)
    # print(df_combined)

    # Perhitungan pertama: Split dataset 70:30
    print("\n--- Perhitungan dengan Split Dataset 70:30 ---")
    train_df, val_df = train_test_split(df_combined, test_size=0.3, shuffle=False)

    flrg, intervals, interval_medians, k = calculate_flrg(train_df)
    # print("\nNilai FLRG:")
    # for key in sorted(flrg.keys()):
    #     print(f"{key} -> {flrg[key]}")

    # Prediksi untuk set validasi (30% data)
    predicted_prices = []
    for i in range(len(val_df)):
        current_price = val_df.iloc[i]['Harga']
        predicted_price = predict_next_price(current_price, intervals, interval_medians, flrg, k)
        predicted_prices.append(predicted_price)

    val_df['Predicted'] = predicted_prices

    # Menghitung MAPE untuk set validasi (30% data)
    mape = np.mean(np.abs((val_df['Harga'] - val_df['Predicted']) / val_df['Harga'])) * 100
    print(f"MAPE pada set validasi (30% data): {mape:.5f}%")

    # Menampilkan beberapa prediksi
    print("\nBeberapa prediksi pada set validasi (30% data):")
    print(val_df[['Date', 'Harga', 'Predicted']].head())

    # Perhitungan kedua: Menggunakan file validasi.csv
    print("\n--- Perhitungan dengan File Validasi Eksternal ---")
    
    # Menggunakan seluruh data untuk training
    flrg, intervals, interval_medians, k = calculate_flrg(df_combined)
    
    # Membaca data validasi
    external_val_df = pd.read_csv(validation_file_path)

    # Prediksi untuk set validasi eksternal
    predicted_prices = []
    for price in external_val_df['Input']:
        predicted_price = predict_next_price(price, intervals, interval_medians, flrg, k)
        predicted_prices.append(predicted_price)

    external_val_df['Predicted'] = predicted_prices

    # Menghitung MAPE untuk set validasi eksternal
    mape = np.mean(np.abs((external_val_df['Real_values'] - external_val_df['Predicted']) / external_val_df['Real_values'])) * 100
    print(f"MAPE pada set validasi eksternal: {mape:.5f}%")

    # Menampilkan beberapa prediksi
    print("\nBeberapa prediksi pada set validasi eksternal:")
    print(external_val_df.head())

    # Opsi: Menampilkan statistik tambahan untuk validasi eksternal
    mae = np.mean(np.abs(external_val_df['Real_values'] - external_val_df['Predicted']))
    rmse = np.sqrt(np.mean((external_val_df['Real_values'] - external_val_df['Predicted'])**2))
    print(f"\nStatistik tambahan untuk validasi eksternal:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")


    # # Print input file predicted prices
    # for i, price in enumerate(predicted_prices):
    #     print(f"Harga: {df_combined.iloc[i]['Harga']}, Prediksi: {price}")

    # # Print user input predicted prices
    # print("Input your price: ")
    # price = float(input())
    # predicted_price = predict_next_price(price, intervals, interval_medians, flrg, k)
    # print(f"Predicted price: {predicted_price}")


    

if __name__ == "__main__":
    main()