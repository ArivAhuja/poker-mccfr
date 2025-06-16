import pandas as pd
import numpy as np

def process_large_csv(input_file, output_file, group_size=1080):
    """
    Process a large CSV file by averaging every group_size rows.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        group_size: Number of rows to average together (default: 1080)
    """
    # Initialize variables
    averages = []
    current_group = []
    row_count = 0
    
    # Read the CSV file in chunks to handle large files efficiently
    chunk_size = 100000  # Process 100k rows at a time
    
    print(f"Processing {input_file}...")
    
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, header=None):
        # Convert chunk to numpy array for faster processing
        values = chunk.values.flatten()
        
        for value in values:
            current_group.append(value)
            row_count += 1
            
            # When we've collected group_size values, calculate average
            if len(current_group) == group_size:
                avg = np.mean(current_group)
                averages.append(avg)
                current_group = []
                
                # Print progress every 10 million rows
                if row_count % 10000000 == 0:
                    print(f"Processed {row_count:,} rows, created {len(averages):,} averages")
    
    # Handle any remaining values in the last incomplete group
    if current_group:
        avg = np.mean(current_group)
        averages.append(avg)
    
    print(f"Total rows processed: {row_count:,}")
    print(f"Total averages created: {len(averages):,}")
    
    # Save the averages to the output file
    print(f"Saving results to {output_file}...")
    output_df = pd.DataFrame(averages, columns=['average'])
    output_df.to_csv(output_file, index=False, header=False)
    
    print("Done!")
    return len(averages)

# Main execution
if __name__ == "__main__":
    input_file = "abstraction_c/data/equity.csv"
    output_file = "abstraction_c/data/equity.csv"
    group_size = 1080
    
    # Process the file
    num_averages = process_large_csv(input_file, output_file, group_size)
    
    print(f"\nProcessing complete!")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Group size: {group_size}")
    print(f"Number of averages written: {num_averages:,}")