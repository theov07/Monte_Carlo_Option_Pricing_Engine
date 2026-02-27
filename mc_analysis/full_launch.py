"""full_launch.py -- Master launcher for mc_analysis."""
import sys, os, subprocess, time

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(BASE_DIR, "..")

SCRIPTS = [
    ("Verification MC math",             "verify_mc.py"),
    ("Battery of component tests",       "test_mc_battery.py"),
    ("MC vs Black-Scholes",              "compare_mc_bs.py"),
    ("Antithetic vs BS",                 "compare_mc_vs_bs_antithetic.py"),
    ("Scalar vs Vectorized european",    "test_scalar_vs_vectorized.py"),
    ("American MC tests",                "test_american.py"),
    ("LS vs Trinomial Tree",             "test_ls_vs_tree.py"),
    ("Trinomial convergence to BS",      "test_convergence.py"),
    ("MC analysis plots",                "plot_mc_analysis.py"),
    ("Trinomial tree timing",            "plot_generator.py"),
]

def run_script(label, filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.isfile(path):
        print(f"  [SKIP] {filename}")
        return False, 0.0
    print(f"\n  >> {label}\n     {filename}\n  " + "-"*74)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, path], cwd=PARENT_DIR)
    elapsed = time.perf_counter() - t0
    ok = r.returncode == 0
    print(f"\n  {'[OK]' if ok else '[FAIL] rc='+str(r.returncode)}  ({elapsed:.1f}s)")
    return ok, elapsed

def main():
    print("="*78 + "\n  MC ANALYSIS -- FULL LAUNCH\n" + "="*78)
    print(f"  Project dir : {PARENT_DIR}\n  Scripts     : {len(SCRIPTS)}\n")
    results = []
    for label, filename in SCRIPTS:
        ok, elapsed = run_script(label, filename)
        results.append((label, filename, ok, elapsed))
    total = sum(e for *_,e in results)
    n_ok = sum(1 for *_,ok,_ in results if ok)
    n_fail = len(results) - n_ok
    print("\n" + "="*78 + "\n  FINAL SUMMARY\n" + "="*78)
    for label, filename, ok, elapsed in results:
        print(f"  {'[OK  ]' if ok else '[FAIL]'}  {elapsed:6.1f}s  {filename:<42}  {label}")
    print("-"*78)
    print(f"  {n_ok}/{len(results)} OK  --  total: {total:.1f}s")
    print("  ALL PASSED" if n_fail==0 else f"  {n_fail} FAILED")
    print("="*78)
    plots_dir = os.path.join(BASE_DIR, "plots")
    if os.path.isdir(plots_dir):
        pngs = sorted(f for f in os.listdir(plots_dir) if f.endswith(".png"))
        if pngs:
            print(f"\n  {len(pngs)} plot(s) in mc_analysis/plots/:")
            for p in pngs: print(f"    - {p}")
    sys.exit(0 if n_fail==0 else 1)

if __name__ == "__main__":
    main()
