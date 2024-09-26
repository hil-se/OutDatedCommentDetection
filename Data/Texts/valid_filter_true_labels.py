import csv

# Input and output file paths
input_file = './java_valid_all.csv'
output_file = './java_valid.csv'

# Open the input CSV file and the output CSV file
with open(input_file, 'r') as csv_in_file, open(output_file, 'w', newline='') as csv_out_file:
    csv_reader = csv.reader(csv_in_file)
    csv_writer = csv.writer(csv_out_file)

    # Get the header
    header = next(csv_reader)
    # Write the header to the output CSV
    csv_writer.writerow(header)

    # Get the index of the 'Label' column
    label_index = header.index('Label')

    # Process each row
    for row in csv_reader:
        # Check if the 'Label' column is 'True' or 'true'
        if row[label_index] in ['True', 'true']:
            # Write the row to the output CSV
            csv_writer.writerow(row)
