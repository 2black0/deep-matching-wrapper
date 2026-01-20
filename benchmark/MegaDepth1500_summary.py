import os
import csv

def parse_results_file(file_path):
    """
    Parses results.txt and returns a dictionary of metrics.
    """
    data = {}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            # 1. Get Matcher Name
            if line.startswith("Matcher:"):
                # Example: Matcher: aliked-lightglue (Outlier: cv-magsac)
                # We want "aliked-lightglue"
                content = line.split(":", 1)[1].strip()
                if "(" in content:
                    matcher_name = content.split("(")[0].strip()
                else:
                    matcher_name = content
                data["Matcher"] = matcher_name
                
            # 2. Get AUC Metrics
            # Example: "AUC @  5 deg: 60.79"
            elif line.startswith("AUC @"):
                parts = line.split(":")
                if len(parts) == 2:
                    key_raw = parts[0].strip()
                    value_raw = parts[1].strip()
                    
                    if "5 deg" in key_raw:
                        data["AUC@5"] = value_raw
                    elif "10 deg" in key_raw:
                        data["AUC@10"] = value_raw
                    elif "20 deg" in key_raw:
                        data["AUC@20"] = value_raw
                        
            # 3. Get Time Statistics
            # Example: "Min: 50.82 ms"
            elif line.startswith("Min:") and "ms" in line:
                val = line.split(":")[1].replace("ms", "").strip()
                data["Min Time"] = val
            elif line.startswith("Max:") and "ms" in line:
                val = line.split(":")[1].replace("ms", "").strip()
                data["Max Time"] = val
            elif line.startswith("Avg:") and "ms" in line:
                val = line.split(":")[1].replace("ms", "").strip()
                data["Avg Time"] = val

        return data
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    # --- PATH CONFIGURATION ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    output_csv_path = os.path.join(base_dir, 'MegaDepth1500_summary.csv')
    
    print(f"Script Location : {base_dir}")
    print(f"Scanning Target : {os.path.abspath(outputs_dir)}")
    print(f"Output File     : {output_csv_path}")
    print("-" * 50)

    # Define Header
    headers = [
        "Matcher",
        "AUC@5", "AUC@10", "AUC@20",
        "Min Time", "Max Time", "Avg Time"
    ]
    
    extracted_rows = []
    
    if not os.path.exists(outputs_dir):
        print(f"ERROR: Directory '{outputs_dir}' not found!")
        return

    items = sorted(os.listdir(outputs_dir))
    
    for item in items:
        # Filter for megadepth folders or just check all results.txt?
        # User request says: "data2 di outputs/megadepth-*/results.txt"
        if not item.startswith("megadepth-"):
            continue
            
        item_path = os.path.join(outputs_dir, item)
        
        if os.path.isdir(item_path):
            result_file = os.path.join(item_path, "results.txt")
            
            if os.path.exists(result_file):
                print(f"Processing: {item}")
                row_data = parse_results_file(result_file)
                
                if row_data:
                    ordered_row = []
                    for h in headers:
                        ordered_row.append(row_data.get(h, ""))
                    extracted_rows.append(ordered_row)

    # Write CSV
    if extracted_rows:
        print(f"\nWriting {len(extracted_rows)} rows to CSV...")
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(extracted_rows)
        print(f"Success! File saved at: {output_csv_path}")
    else:
        print("No megadepth results found.")

if __name__ == "__main__":
    main()
