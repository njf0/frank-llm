import csv
import random
import subprocess
from pathlib import Path

files = sorted(list(Path('plan', 'outputs').iterdir()))
most_recent = files[-2]

with open(most_recent, 'r', encoding='utf-8') as f:
    # skip first row
    f.readline()
    rows = list(csv.reader(f))

    while True:

        subprocess.run(['clear'], check=True)

        row = random.choice(rows)

        print(f"Question: {row[0]}")
        print("---")
        print(row[1])
        input()