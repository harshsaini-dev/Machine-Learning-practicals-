"""Run all five practical scripts without opening plot windows."""
from pathlib import Path
import os
import runpy

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parent
os.chdir(project)
# Each experiment already saves its plot before calling show().
plt.show = lambda *args, **kwargs: plt.close("all")
for number in range(1, 6):
    print(f"\nRunning Lab Sheet {number:02d}")
    runpy.run_path(str(project / f"lab_{number:02d}.py"),
                   run_name="__main__")
    plt.close("all")
print("\nAll five lab sheets completed.")
