import csv
import json

# Input and output file paths
input_file = './Java_train.jsonl'
output_file = './java_train.csv'

# Open the input JSONL file and the output CSV file
with open(input_file, 'r') as jsonl_file, open(output_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header to the CSV file
    csv_writer.writerow(['Source', 'Target'])

    # Process each line in the JSONL file
    for line in jsonl_file:
        data = json.loads(line.strip())

        # Write the src_desc and src_method in one row
        csv_writer.writerow([data['src_desc'], data['src_method']])

        # Write the dst_desc and dst_method in a separate row
        csv_writer.writerow([data['dst_desc'], data['dst_method']])