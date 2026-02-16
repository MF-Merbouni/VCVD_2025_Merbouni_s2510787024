import pandas as pd
import numpy as np
import scipy.constants
import matplotlib.pyplot
import argparse
import sys
import math

# we start by loading and reading the coefficients in the tables from an excel file 

filePath = 'Tyre_coefficients.xlsx'

t2Fy = pd.read_excel(filePath, sheet_name= 'Fy_coeffs')
t2Fx = pd.read_excel(filePath, sheet_name= 'Fx_coeffs')
t3Cmbr = pd.read_excel(filePath, sheet_name= 'Camber_coeffs')

# we construct dictionaries to organize our parameters and values for later use 
Fy_coeffs = dict(zip(t2Fy['Parameter'], t2Fy['Value']))
Fx_coeffs = dict(zip(t2Fx['Parameter'], t2Fx['Value']))
Camber_coeffs = dict(zip(t3Cmbr['Parameter'], t3Cmbr['Value']))

# we use the argument parser to handle slip, weight and mu
parser = argparse.ArgumentParser(description="Tyre Force Calculation")
parser.add_argument('slip', type=float, help="Longitudinal slip in degrees")
parser.add_argument('weight', type=float, help="Total vehicle weight in kg")
parser.add_argument('mu', type=float, help="Friction coefficient")

args = parser.parse_args()

# we then calculate Fz per wheel (assuming we have 4 wheels) and weight in kg converted to Newtons using parser
total_weight_n = args.weight * scipy.constants.g
fz = total_weight_n / 4


# we use the following function/method to calculate the brake/longitudinal Force (Fx)

def calculate_fx(load_kn=0.0, long_slip=0.0, coeff=None):
    # Formulas for D, C, B, E
    D_PeakFactor = coeff['a1'] * (load_kn**2) + coeff['a2'] * load_kn
    C_ShapeFactor = 1.65
    
    numerator_B = coeff['a3'] * (load_kn**2) + coeff['a4'] * load_kn
    denominator_B = C_ShapeFactor * D_PeakFactor * np.exp(coeff['a5'] * load_kn)

    B_StiffnessFactor = numerator_B / denominator_B
    
    E_CurvatureFactor = coeff['a6'] * (load_kn**2) + coeff['a7'] * load_kn + coeff['a8']
    
    # Calculate Phi 
    phi_val = (1 - E_CurvatureFactor) * long_slip + (E_CurvatureFactor / B_StiffnessFactor) * np.arctan(B_StiffnessFactor * long_slip)
    
    # Final Fx equation
    brake_force = D_PeakFactor * np.sin(C_ShapeFactor * np.arctan(B_StiffnessFactor * phi_val))
    return brake_force

