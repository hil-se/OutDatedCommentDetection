import csv
import json

# Input and output file paths
input_file = './Java_valid.jsonl'
output_file = './java_valid.csv'

# Open the input JSONL file and the output CSV file
with open(input_file, 'r') as jsonl_file, open(output_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header to the CSV file
    csv_writer.writerow(['Source_Old', 'Target_Old', 'Source_New', 'Target_New', 'Label'])

    # Process each line in the JSONL file
    for line in jsonl_file:
        data = json.loads(line.strip())

        # Write the src_desc and src_method in one row
        csv_writer.writerow([data['src_desc'], data['src_method'], data['dst_desc'], data['dst_method'], data['label']])
