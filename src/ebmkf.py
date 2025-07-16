"""
Minimal 2-layer Energy-Balance Model + manual Extended Kalman Filter
(Replicates Nicklas et al. 2015 Fig 2)

Run from the repo root:
    python -m src.ebmkf data/processed/annual_climate_inputs.csv
"""

import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONSTANTS ──────────────────────────────────────────────────────────────
SECS_PER_YEAR = 365.25 * 24 * 3600

C_S   = 7.00e8      # J m-2 K-1  surface layer (86 m)
C_D   = 1.10e9      # deep-ocean layer (1141 m)
KAPPA = 0.67        # W m-2 K-1  vertical heat exchange
LAMBDA = 1.05       # W m-2 K-1  climate feedback

Q = np.diag([0.12**2, 0.05**2])          # process-noise cov (K²)
R = np.diag([0.05**2, (0.8e22 / C_D)**2])  # obs-error cov   (K²)
H = np.eye(2)                            # observation matrix

# ── MODEL FUNCTIONS ────────────────────────────────────────────────────────
def f(x, F_t):
    """One-year step of the 2-layer energy-balance model."""
    T_s, T_d = x
    dT_s = (F_t - LAMBDA*T_s - KAPPA*(T_s - T_d)) / C_S
    dT_d = (              KAPPA*(T_s - T_d))      / C_D
    return np.array([T_s + dT_s*SECS_PER_YEAR,
                     T_d + dT_d*SECS_PER_YEAR])

def jacobian(x, _F_t):
    """Jacobian ∂f/∂x (discrete-time)."""
    a = (LAMBDA + KAPPA)*SECS_PER_YEAR / C_S
    b =  KAPPA           *SECS_PER_YEAR / C_S
    c =  KAPPA           *SECS_PER_YEAR / C_D
    d =  KAPPA           *SECS_PER_YEAR / C_D
    return np.array([[1 - a,  b],
                     [   c , 1 - d]])

# ── MAIN FILTER ────────────────────────────────────────────────────────────
def run_filter(df):
    """Extended Kalman Filter through the annual table."""
    x  = np.zeros(2)                         # initial state
    P  = np.diag([0.30**2, 0.30**2])         # initial covariance
    xs, sigmas = [], []

    for F_t, gmst, ohc in zip(df.erf_wm2, df.gmst_obs, df.OHC):
        # ------ predict ----------------------------------------------------
        F_k  = jacobian(x, F_t)
        x_pr = f(x, F_t)
        P_pr = F_k @ P @ F_k.T + Q

        # ------ update -----------------------------------------------------
        z  = np.array([gmst, ohc / C_D])     # OHC (J) ➜ °C
        y  = z - H @ x_pr                    # innovation
        S  = H @ P_pr @ H.T + R
        K  = P_pr @ H.T @ np.linalg.inv(S)   # Kalman gain
        x  = x_pr + K @ y                    # posterior mean
        P  = (np.eye(2) - K @ H) @ P_pr      # posterior cov

        xs.append(x.copy())
        sigmas.append(np.sqrt(P[0,0]))

    out = df.copy()
    out[["T_s", "T_d"]] = np.vstack(xs)
    out["sigma"]        = sigmas
    return out

# ── CLI QUICK-LOOK ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, pathlib, matplotlib.pyplot as plt, matplotlib

    if len(sys.argv) not in (2, 3):
        print("Usage: python -m src.ebmkf <annual_csv> [output_png]")
        sys.exit(1)

    csv_path = pathlib.Path(sys.argv[1])
    out_png  = pathlib.Path(sys.argv[2]) if len(sys.argv) == 3 else \
               pathlib.Path("fig2_replica.png")

    df  = pd.read_csv(csv_path)
    out = run_filter(df)

    plt.figure(figsize=(9, 4))
    plt.fill_between(out.year, out.T_s - out.sigma, out.T_s + out.sigma,
                     alpha=.3, label="±1 σ (state)")
    plt.plot(out.year, out.T_s, label="EBM-KF state")
    plt.scatter(out.year, out.gmst_obs, s=10, c="k", alpha=.4,
                label="HadCRUT5 obs")
    plt.ylabel("°C anomaly  (1961-1990 base)")
    plt.xlabel("Year")
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(out_png, dpi=200)
    print(f"✓  Figure saved to {out_png.resolve()}")

    # show only if backend is interactive
    if matplotlib.get_backend() not in ("agg", "pdf", "svg"):
        plt.show()
