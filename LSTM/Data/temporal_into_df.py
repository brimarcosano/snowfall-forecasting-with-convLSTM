import pandas as pd
import os
import matplotlib.pyplot as plt

def process_weekly_data(excel_files_directory):
    desired_columns = [
        'STATION', 'County', 'LATITUDE', 'LONGITUDE',
        'ELEVATION', 'DATE', 'AWND', 'PRCP', 'SNOW', 'SNWD',
        'TAVG', 'TMAX', 'TMIN', 'vpdmin (hPa)', 'vpdmax (hPa)'
    ]

   # Read and combine data
    dataframes = []
    for filename in os.listdir(excel_files_directory):
        if filename.endswith('.xlsx') and not filename.startswith('~$'):
            file_path = os.path.join(excel_files_directory, filename)
            try:
                df = pd.read_excel(file_path, engine='openpyxl', usecols=desired_columns)
                # available_cols = [col for col in desired_columns if col in df.columns]
                # df = df[available_cols]
                if not df.empty:
                    dataframes.append(df)
                    print(f"Columns in {filename}: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")

    if not dataframes:
        raise ValueError("No valid data found in any files")

    final_df = pd.concat(dataframes)
    print("\nColumns in combined data:")
    print(final_df.columns.tolist())

    # Convert DATE to datetime and set as index
    final_df['DATE'] = pd.to_datetime(final_df['DATE'])
    
    # Calculate freezing days
    final_df['freezing_day'] = (final_df['TMIN'] <= 32).astype(int)

    # Group by week
    weekly_grouped = final_df.groupby(['STATION', 'County', 'LATITUDE', 'LONGITUDE', 'ELEVATION'])

    # Aggregate weekly data
    weekly_data = weekly_grouped.resample('W', on='DATE').agg({
        'SNOW': 'sum',
        'SNWD': 'max',
        'PRCP': 'sum',
        'TMAX': ['max', 'mean'],
        'TMIN': ['min', 'mean'],
        'TAVG': 'mean',
        'AWND': 'mean',
        'vpdmin (hPa)': 'mean',
        'vpdmax (hPa)': 'mean',
        'freezing_day': 'sum'
    }).reset_index()

    # Flatten column names
    weekly_data.columns = [''.join(col).strip() if isinstance(col, tuple) else col for col in weekly_data.columns]

    # Save processed data
    output_path = 'CNN_LSTM_Thesis/LSTM/Data/weekly_processed_data.csv'
    weekly_data.to_csv(output_path, index=False)
    
    return weekly_data

if __name__ == "__main__":
    excel_dir = '/Users/briannamarcosano/Documents/Thesis/Data/Temp Data/Good Data'
    try:
        weekly_data = process_weekly_data(excel_dir)
    except Exception as e:
        print(f"Error processing data: {str(e)}")