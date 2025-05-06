import pandas as pd

def clean_test_data(file_path):
    data = pd.read_csv(file_path)

    # Fill missing 'Risk Classification' values with the most common class
    if data['Risk Classification'].isnull().any():
        data['Risk Classification'].fillna(data['Risk Classification'].mode()[0], inplace=True)

    output_file = file_path.replace(".csv", "_cleaned.csv")
    data.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

# Run the cleaning script
clean_test_data("../../../data/ver_2/output_files/dataset_2/test_data.csv")

