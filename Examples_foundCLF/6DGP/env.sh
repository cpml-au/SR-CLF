# Knobs of the 6-D CLF/CMA run -- source this from a job script (every value is the default of
# the 4-D V8 pipeline unless the comment says otherwise).
# --- the system and the search space ---------------------------------------
export SYMCLF6D_SYSTEM="${SYMCLF6D_SYSTEM:-quad6d}"     # quad6d | cartpole4d (acceptance)
export SYMCLF6D_TEMPLATE="${SYMCLF6D_TEMPLATE:-full}"   # full = 21 coefficients | are = 9
export SYMCLF6D_TEMPLATE_X0="${SYMCLF6D_TEMPLATE_X0:-1.0}"   # no LQR initialisation
export SYMCLF_V2_BOX="${SYMCLF_V2_BOX:-3,3,3,3,3,3}"    # the paper's D = {|x_i| <= 3}
export SYMCLF6D_GRID_POINTS="${SYMCLF6D_GRID_POINTS:-7}"     # training grid 7^6 = 117,649
# DET4 scan lattice: mesh^6 + 6*lines^5*samples + 200*401 = 1,365,707 points
export SYMCLF6D_LATTICE_MESH="${SYMCLF6D_LATTICE_MESH:-7}"
export SYMCLF6D_LATTICE_LINES="${SYMCLF6D_LATTICE_LINES:-3}"
export SYMCLF6D_LATTICE_SAMPLES="${SYMCLF6D_LATTICE_SAMPLES:-801}"
# --- DET4 (V8 job values) ---------------------------------------------------
export SYMCLF_DET4_RHO="${SYMCLF_DET4_RHO:-1000}"       # per-input bound |u_i| <= rho
export SYMCLF6D_B_NORM="${SYMCLF6D_B_NORM:-l1}"         # l1 = that per-input box | l2 = ball
export SYMCLF_DET4_PD_EPS="${SYMCLF_DET4_PD_EPS:-1e-4}"
export SYMCLF_DET4_KAPPA_MIN="${SYMCLF_DET4_KAPPA_MIN:-0}"
export SYMCLF_DET4_GATE_KAPPA_CAP="${SYMCLF_DET4_GATE_KAPPA_CAP:-1e6}"
export SYMCLF_DET4_PROJECT_ASCENT="${SYMCLF_DET4_PROJECT_ASCENT:-1}"
export SYMCLF_DET4_ORIGIN_SEEDS="${SYMCLF_DET4_ORIGIN_SEEDS:-1}"
export SYMCLF_DET4_B_REL_TOL="${SYMCLF_DET4_B_REL_TOL:-1e-6}"
# --- the price rule ---------------------------------------------------------
export SYMCLF_CP3D_ROA_WEIGHT="${SYMCLF_CP3D_ROA_WEIGHT:-500}"
export SYMCLF_CP3D_ROA_C_TARGET="${SYMCLF_CP3D_ROA_C_TARGET:-0.004}"
export SYMCLF_DET4_PEN_CV="${SYMCLF_DET4_PEN_CV:-10}"
export SYMCLF_DET4_PEN_CV_TARGET="${SYMCLF_DET4_PEN_CV_TARGET:-0.05}"
export SYMCLF_DET4_PEN_KAPPA="${SYMCLF_DET4_PEN_KAPPA:-30}"
export SYMCLF_DET4_PEN_KAPPA_TARGET="${SYMCLF_DET4_PEN_KAPPA_TARGET:-0.05}"
export SYMCLF_DET4_PEN_VOL="${SYMCLF_DET4_PEN_VOL:-20}"
export SYMCLF_DET4_PEN_VOL_TARGET="${SYMCLF_DET4_PEN_VOL_TARGET:-0.02}"
export SYMCLF_LENGTH_REG="${SYMCLF_LENGTH_REG:-0.005}"
# --- threading --------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
