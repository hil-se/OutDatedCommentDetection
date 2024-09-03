import csv
import json

# Input and output file paths
input_file = './Java_test.jsonl'
output_file = './java_test.csv'

# Open the input JSONL file and the output CSV file
with open(input_file, 'r') as jsonl_file, open(output_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header to the CSV file
    csv_writer.writerow(['Source', 'Target', 'Label'])

    # Process each line in the JSONL file
    for line in jsonl_file:
        data = json.loads(line.strip())

        # Write src_desc, dst_method, and label to the CSV file
        csv_writer.writerow([data['src_desc'], data['dst_method'], data['label']])

