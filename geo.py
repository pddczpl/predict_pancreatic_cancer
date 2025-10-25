import gzip
import shutil
from pathlib import Path

folder = Path(r"C:\Users\DatGo\OneDrive\Documents\Personal_Project\predict_pancreatic_cancer\geo\GSE62452_RAW")

for gz_file in folder.glob("*.gz"):
    out_file = gz_file.with_suffix('')  # bỏ đuôi .gz
    with gzip.open(gz_file, 'rb') as f_in:
        with open(out_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extracted {gz_file} -> {out_file}")
