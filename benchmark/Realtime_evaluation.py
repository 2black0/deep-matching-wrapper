import os
import pandas as pd
import argparse
from pathlib import Path

def main():
    # Configuration
    base_dir = Path(__file__).parent.parent
    outputs_dir = base_dir / "outputs" / "realtime"
    docs_dir = base_dir / "docs"
    output_md_path = docs_dir / "REALTIME_RESULT.md"
    
    # Ensure output directory exists
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning directory: {outputs_dir}")
    
    if not outputs_dir.exists():
        print(f"Error: Directory {outputs_dir} does not exist.")
        return

    results = []

    # Iterate over subdirectories in outputs/realtime/
    for matcher_dir in sorted(outputs_dir.iterdir()):
        if matcher_dir.is_dir():
            result_csv = matcher_dir / "result.csv"
            
            if result_csv.exists():
                matcher_name = matcher_dir.name
                print(f"Processing {matcher_name}...")
                
                try:
                    df = pd.read_csv(result_csv)
                    
                    # Skip first 5 rows (warmup)
                    if len(df) > 5:
                        df_subset = df.iloc[5:]
                    else:
                        print(f"  Warning: Less than 5 rows for {matcher_name}, using all available data.")
                        df_subset = df
                    
                    if not df_subset.empty:
                        avg_metrics = df_subset.mean(numeric_only=True)
                        
                        results.append({
                            "Matcher": matcher_name,
                            "Total Kpts0": round(avg_metrics.get("total_kpts0", 0), 1),
                            "Total Kpts1": round(avg_metrics.get("total_kpts1", 0), 1),
                            "Matched": round(avg_metrics.get("matched_kpts", 0), 1),
                            "Inliers": round(avg_metrics.get("inliers", 0), 1),
                            "FPS": round(avg_metrics.get("fps", 0), 2),
                            "GPU Util (%)": round(avg_metrics.get("gpu_usage_%", 0), 2),
                            "GPU Mem (MB)": round(avg_metrics.get("gpu_memory_mb", 0), 2),
                            "CPU Util (%)": round(avg_metrics.get("cpu_usage_%", 0), 2),
                            "Mem (MB)": round(avg_metrics.get("memory_mb", 0), 2)
                        })
                    else:
                         print(f"  Warning: No data available for {matcher_name} after skipping warmup.")
                
                except Exception as e:
                    print(f"  Error reading CSV for {matcher_name}: {e}")

    # Create DataFrame for results
    if results:
        results_df = pd.DataFrame(results)
        
        # Format table columns
        cols = ["Matcher", "Total Kpts0", "Total Kpts1", "Matched", "Inliers", "FPS", 
                "GPU Util (%)", "GPU Mem (MB)", "CPU Util (%)", "Mem (MB)"]
        
        # Ensure all columns exist
        for col in cols:
            if col not in results_df.columns:
                results_df[col] = "N/A"
                
        results_df = results_df[cols]
        
        # Generate Markdown
        md_content = "# Realtime Matcher Evaluation Results\n\n"
        md_content += f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md_content += "Note: Averages calculated excluding the first 5 frames (warmup).\n\n"
        md_content += results_df.to_markdown(index=False)
        
        # Write to file
        with open(output_md_path, "w") as f:
            f.write(md_content)
            
        print(f"\nSuccessfully generated report at: {output_md_path}")
        print("\nPreview:")
        print(results_df.to_markdown(index=False))
        
    else:
        print("No valid results found to generate report.")

if __name__ == "__main__":
    main()
