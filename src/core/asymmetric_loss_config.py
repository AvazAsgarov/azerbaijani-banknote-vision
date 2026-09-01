"""Asymmetric risk penalty config (beta_c) for AZN-Vision loss function."""
BETA_C = {0: 1.0, 1: 1.2, 2: 1.5, 3: 1.8, 4: 2.5, 5: 3.5, 6: 5.0}
_total = sum(BETA_C.values())
_n = len(BETA_C)
BETA_C_NORMALISED = {k: v * _n / _total for k, v in BETA_C.items()}
CLASS_NAMES = ["1 AZN", "5 AZN", "10 AZN", "20 AZN", "50 AZN", "100 AZN", "200 AZN"]
