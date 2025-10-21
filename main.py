"""
batch_runner.py

– Manage parameter sweeps and Monte Carlo runs:
    • vary noise levels, drift, and hardware‐uncertainty parameters
    • run multiple simulation jobs in a loop
– Aggregate results:
    • mean, standard deviation, worst‐case fidelity
– Save results to CSV or JSON for later analysis
"""

import sys
sys.path.append('src')

def main():
    print("Hello from apollo!")


if __name__ == "__main__":
    main()
