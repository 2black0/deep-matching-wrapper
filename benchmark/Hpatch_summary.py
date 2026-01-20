import os
import csv

def parse_results_file(file_path):
    """
    Membaca file results.txt dan mengembalikan dictionary berisi metrics.
    """
    data = {}
    current_section = None 
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            # 1. Ambil nama Matcher
            if line.startswith("Matcher:"):
                # Mengambil teks setelah tanda titik dua
                data["Matcher"] = line.split(":", 1)[1].strip()
                
            # 2. Deteksi Section
            if "Subset: Total" in line:
                current_section = "Total"
            elif "Subset: Illumination" in line:
                current_section = "Illumination"
            elif "Subset: Viewpoint" in line:
                current_section = "Viewpoint"
            elif "Timing (ms):" in line:
                current_section = "Time"
            
            # 3. Parsing Nilai
            if current_section in ["Total", "Illumination", "Viewpoint"]:
                parts = line.split(":")
                if len(parts) == 2:
                    key_raw = parts[0].strip()
                    value_raw = parts[1].strip().replace('%', '') # Hapus %
                    
                    metric_name = None
                    
                    # Mapping nama key text file ke nama header CSV
                    if "AUC@" in key_raw:
                        metric_name = key_raw
                    elif "MHA@" in key_raw:
                        metric_name = key_raw
                    elif "MEAN_MMA@" in key_raw:
                        metric_name = key_raw.replace("MEAN_", "") # Hapus MEAN_
                    
                    if metric_name:
                        full_key = f"{current_section} {metric_name}"
                        data[full_key] = value_raw

            elif current_section == "Time":
                parts = line.split(":")
                if len(parts) == 2:
                    key_raw = parts[0].strip()
                    value_raw = parts[1].strip()
                    if key_raw in ["Min", "Max", "Avg"]:
                        full_key = f"Time {key_raw}"
                        data[full_key] = value_raw

        return data
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    # --- KONFIGURASI PATH ---
    # Mendapatkan lokasi absolut script ini berada (benchmark_tools)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path ke folder outputs (naik satu level '..', lalu masuk 'outputs')
    outputs_dir = os.path.join(base_dir, '..', 'outputs')
    
    # Path untuk file output CSV (disimpan di folder yang sama dengan script ini)
    output_csv_path = os.path.join(base_dir, 'Hpatch_summary.csv')
    
    print(f"Script Location : {base_dir}")
    print(f"Scanning Target : {os.path.abspath(outputs_dir)}")
    print(f"Output File     : {output_csv_path}")
    print("-" * 50)

    # Definisi Header
    headers = [
        "Matcher",
        "Total AUC@1", "Total AUC@3", "Total AUC@5", "Total AUC@10",
        "Total MHA@3", "Total MHA@5", "Total MHA@7",
        "Total MMA@3", "Total MMA@5", "Total MMA@7",
        "Illumination AUC@1", "Illumination AUC@3", "Illumination AUC@5", "Illumination AUC@10",
        "Illumination MHA@3", "Illumination MHA@5", "Illumination MHA@7",
        "Illumination MMA@3", "Illumination MMA@5", "Illumination MMA@7",
        "Viewpoint AUC@1", "Viewpoint AUC@3", "Viewpoint AUC@5", "Viewpoint AUC@10",
        "Viewpoint MHA@3", "Viewpoint MHA@5", "Viewpoint MHA@7",
        "Viewpoint MMA@3", "Viewpoint MMA@5", "Viewpoint MMA@7",
        "Time Min", "Time Max", "Time Avg"
    ]
    
    extracted_rows = []
    
    # Cek apakah folder outputs ada
    if not os.path.exists(outputs_dir):
        print(f"ERROR: Directory '{outputs_dir}' not found!")
        return

    items = sorted(os.listdir(outputs_dir))
    
    for item in items:
        item_path = os.path.join(outputs_dir, item)
        
        # Cek apakah ini direktori
        if os.path.isdir(item_path):
            result_file = os.path.join(item_path, "results.txt")
            
            # Cek keberadaan results.txt
            if os.path.exists(result_file):
                print(f"Processing: {item}")
                row_data = parse_results_file(result_file)
                
                if row_data:
                    ordered_row = []
                    for h in headers:
                        # Ambil data sesuai header, kosongkan jika tidak ada
                        ordered_row.append(row_data.get(h, ""))
                    extracted_rows.append(ordered_row)

    # Tulis CSV
    if extracted_rows:
        print(f"\nWriting {len(extracted_rows)} rows to CSV...")
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(extracted_rows)
        print(f"Success! File saved at: {output_csv_path}")
    else:
        print("No results.txt found in subdirectories.")

if __name__ == "__main__":
    main()