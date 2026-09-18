# list_indices.py
import pandas as pd
import config
from dataset import load_json_dataset

def main():
    output_file = "test_indices_by_class.txt"
    print("Parsing dataset metadata...")
    _, df_test = load_json_dataset(config.ROOT_JSON_PATH)
    
    # Filter for selected objects matching config settings
    df_test = df_test[df_test["category"].isin(config.SELECTED_OBJECTS)].reset_index(drop=True)
    
    print(f"Grouping indices and exporting to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("          REAL-IAD TEST DATASET INDICES BY ANOMALY CLASS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Test Dataset Size: {len(df_test)} samples\n")
        f.write("Use any of the indices listed below in config.VISUALIZATION_SAMPLE_IDX\n\n")
        
        # Group indices by anomaly type
        grouped = df_test.groupby("anomaly_class")
        
        for class_name, group in grouped:
            indices = group.index.tolist()
            total_count = len(indices)
            
            f.write(f"🔹 Class: {class_name} ({total_count} total samples)\n")
            
            # Pretty print the list of indices wrapped neatly at 80 characters wide
            indices_str = ", ".join(map(str, indices))
            wrapped_lines = []
            current_line = "  Indices: "
            
            for idx in map(str, indices):
                if len(current_line) + len(idx) + 2 > 85:
                    wrapped_lines.append(current_line.rstrip(", "))
                    current_line = "           " + idx + ", "
                else:
                    current_line += idx + ", "
            wrapped_lines.append(current_line.rstrip(", "))
            
            f.write("\n".join(wrapped_lines) + "\n")
            f.write("-" * 70 + "\n\n")
            
    print(f"Done! You can now open '{output_file}' to view all indices.")

if __name__ == "__main__":
    main()