import argparse
from pathlib import Path
import yaml
import json
from sigma_summary import SigmaSummaryGenerator

def main():
    parser = argparse.ArgumentParser(description="Batch process Sigma YAML files to base log JSONs per technique.")
    parser.add_argument('-i', '--input_dir', required=True, help='Folder containing YAMLs (recursively).')
    parser.add_argument('-o', '--output_dir', required=True, help='Folder to save base logs by technique.')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir_sub = output_dir / "sub_variants"  # Directory for sub-variants
    yaml_files = list(input_dir.glob("**/*.yml")) + list(input_dir.glob("**/*.yaml"))

    for yaml_path in yaml_files:
        try:
            # For each YAML file, process it
            print(f"Processing {yaml_path}")
            generator = SigmaSummaryGenerator(str(yaml_path), "dummy.json")
            base_logs = generator.generate_base_logs_from_condition()

            for i, base_log in enumerate(base_logs):
                techniques = base_log.get('technique', [])
                if not techniques:
                    techniques = ["unknown"]
                for tech in techniques:
                    tech_dir = output_dir / tech
                    tech_dir_sub = output_dir_sub / tech  # Ensure the path is absolute
                    tech_dir.mkdir(parents=True, exist_ok=True)
                    tech_dir_sub.mkdir(parents=True, exist_ok=True)
                    # Save base log
                    out_file = tech_dir / f"{yaml_path.stem}_variant_{i}.json"
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(base_log, f, indent=2)
                    # Generate and save sub-variants
                    sub_variants = generator.generate_sub_variants(base_log)
                    for j, sub in enumerate(sub_variants):
                        sub_file = tech_dir_sub / f"{yaml_path.stem}_variant_{i}_sub_{j}.json"
                        with open(sub_file, "w", encoding="utf-8") as sf:
                            json.dump(sub, sf, indent=2)
            print(f"Done: {yaml_path}")
        except Exception as e:
            print(f"ERROR processing {yaml_path}: {e}")

if __name__ == "__main__":
    main()
