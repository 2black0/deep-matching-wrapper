import os
import pandas as pd
import re
from pathlib import Path

def parse_result_txt(file_path):
    """Parses result.txt to extract metrics."""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Extract metrics using regex
        # Pattern looks for "Label: Value (type)"
        patterns = {
            "Total Kpts0": r"Total Keypoints0:\s+(\d+)",
            "Total Kpts1": r"Total Keypoints1:\s+(\d+)",
            "Matched": r"Matched Keypoints:\s+(\d+)",
            "Inliers": r"Inliers:\s+(\d+)", 
            "Ratio": r"Ratio:\s+([\d\.]+)",
            "Time (ms)": r"Time:\s+([\d\.]+)\s+ms"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                val = match.group(1)
                # Convert to int or float
                if "." in val:
                    metrics[key] = float(val)
                else:
                    metrics[key] = int(val)
            else:
                metrics[key] = None
                
        return metrics
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def main():
    # Configuration
    base_dir = Path(__file__).parent.parent
    outputs_dir = base_dir / "outputs" / "matching"
    docs_dir = base_dir / "docs"
    output_md_path = docs_dir / "MATCHING_RESULT.md"
    
    # Ensure output directory exists
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning directory: {outputs_dir}")
    
    if not outputs_dir.exists():
        print(f"Error: Directory {outputs_dir} does not exist.")
        return

    results = []

    # Iterate over subdirectories in outputs/matching/
    if outputs_dir.exists():
        for matcher_dir in sorted(outputs_dir.iterdir()):
            if matcher_dir.is_dir():
                result_txt = matcher_dir / "result.txt"
                
                if result_txt.exists():
                    # Parse directory name to get matcher name
                    # Format: {matcher}_{stem1}_{stem2}
                    # We assume the first part is matcher, but matcher names can have underscores (e.g. sift_lightglue)
                    # A safer way might be to just use the whole folder name or try to be smart if needed.
                    # For now using folder name is safest to identify the run.
                    run_name = matcher_dir.name
                    
                    print(f"Processing {run_name}...")
                    
                    metrics = parse_result_txt(result_txt)
                    
                    if metrics:
                        row = {"Run": run_name}
                        row.update(metrics)
                        results.append(row)
                    else:
                         print(f"  Warning: Could not parse results for {run_name}")

    # Create DataFrame for results
    if results:
        results_df = pd.DataFrame(results)
        
        # Format table columns
        cols = ["Run", "Total Kpts0", "Total Kpts1", "Matched", "Inliers", "Ratio", "Time (ms)"]
        
        # Ensure all columns exist
        for col in cols:
            if col not in results_df.columns:
                results_df[col] = "N/A"
                
        results_df = results_df[cols]
        
        # Generate Markdown
        md_content = "# Matching Evaluation Results\n\n"
        md_content += f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
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
