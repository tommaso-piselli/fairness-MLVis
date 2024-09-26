import os
import csv
import glob

# Specify the folder path containing the CSV files
folder_path = r'.\output\csv\AVN\region'

# Specify the output file name
output_file = r'.\output\csv\AVN\AVN_region.csv'

# Get a list of all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Check if there are any CSV files in the folder
if not csv_files:
    print("No CSV files found in the specified folder.")
else:
    # Open the output file in write mode
    with open(output_file, 'w', newline='') as outfile:
        # Assume all files have the same header, so we'll use the first file's header
        with open(csv_files[0], 'r') as firstfile:
            header = next(csv.reader(firstfile))

        # Create a CSV writer object
        writer = csv.writer(outfile)

        # Write the header to the output file
        writer.writerow(header)

        # Process each CSV file
        for file in csv_files:
            with open(file, 'r') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip the header row
                # Write all rows from this file to the output file
                for row in reader:
                    writer.writerow(row)

    print(f"All CSV files have been combined into {output_file}")
