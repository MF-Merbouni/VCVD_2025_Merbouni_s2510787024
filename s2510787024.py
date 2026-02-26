import pandas as pd
import argparse
import scipy.constants
import numpy as np
import matplotlib.pyplot as plt


# we start by loading and reading the coefficients in the tables from an excel file 

filePath = 'Tyre_coefficients.xlsx'

t2Fy = pd.read_excel(filePath, sheet_name= 'Fy_coeffs')
t2Fx = pd.read_excel(filePath, sheet_name= 'Fx_coeffs')
t3Cmbr = pd.read_excel(filePath, sheet_name= 'Camber_coeffs')

# we construct dictionaries to organize our parameters and values for later use 
Fy_coeffs = dict(zip(t2Fy['Parameter'], t2Fy['Value']))
Fx_coeffs = dict(zip(t2Fx['Parameter'], t2Fx['Value']))
Camber_coeffs = dict(zip(t3Cmbr['Parameter'], t3Cmbr['Value']))

# Merging the side force (Fy) and the camber coefficients in one dictionary
SideForce_coeffs_all = {**Fy_coeffs, **Camber_coeffs}

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

def calculate_fx(load_kn=0.0, long_slip=0.0, coeff=None, friction_mu=1.0):
    # Formulas for D, C, B, E

    # Peak Factor (D) 
    D_PeakFactor = (coeff['a1'] * (load_kn**2) + coeff['a2'] * load_kn) * friction_mu

    # Shape Factor (C)
    C_ShapeFactor = 1.65
    
    # Stiffness Factor (B)
    numerator_B = coeff['a3'] * (load_kn**2) + coeff['a4'] * load_kn
    denominator_B = C_ShapeFactor * D_PeakFactor * np.exp(coeff['a5'] * load_kn)

    B_StiffnessFactor = numerator_B / denominator_B
    
    # Curvature Factor (E)
    E_CurvatureFactor = coeff['a6'] * (load_kn**2) + coeff['a7'] * load_kn + coeff['a8']
    
    # Calculate Phi 
    phi_val = (1 - E_CurvatureFactor) * long_slip + (E_CurvatureFactor / B_StiffnessFactor) * np.arctan(B_StiffnessFactor * long_slip)
    
    # Final Fx equation
    brake_force = D_PeakFactor * np.sin(C_ShapeFactor * np.arctan(B_StiffnessFactor * phi_val))
    return brake_force

# we use the following function/method to calculate the side/lateral Force (Fy)

def calculate_fy(load_kn=0.0, slip_angle=0.0, coeff=None, friction_mu=1.0):
    # camber angle 
    camber_gamma = 0 
    
    # Sh and Sv (Shifts) 
    horizontal_shift = coeff.get('a9', 0) * camber_gamma
    vertical_shift = (coeff.get('a10', 0) * (load_kn**2) + coeff.get('a11', 0) * load_kn) * camber_gamma
    
    # Formulas for D, C, B, E
    
    # Peak Factor (D) 
    D_PeakFactor = ((coeff['a1'] * (load_kn**2)) + (coeff['a2'] * load_kn)) * friction_mu
    
    # Shape Factor (C)
    C_ShapeFactor = 1.30
    
    # Stiffness Factor (B)
    numerator_B = (coeff['a3'] * np.sin(coeff['a4'] * np.arctan(coeff['a5'] * load_kn))) * (1 - coeff['a12'] * np.abs(camber_gamma))
    denominator_B = C_ShapeFactor * D_PeakFactor
    B_StiffnessFactor = numerator_B / denominator_B
    
    # Curvature Factor (E)
    E_CurvatureFactor = (coeff['a6'] * (load_kn**2)) + (coeff['a7'] * load_kn) + (coeff['a8'])
    
    # Side Force
    alpha_star = slip_angle + horizontal_shift
    phi_val = (1 - E_CurvatureFactor) * alpha_star + (E_CurvatureFactor / B_StiffnessFactor) * np.arctan(B_StiffnessFactor * alpha_star)
    
    side_force = D_PeakFactor * np.sin(C_ShapeFactor * np.arctan(B_StiffnessFactor * phi_val)) + vertical_shift
    
    return side_force

# we use the following function to print the final results

def final_plot(vertical_load_n, input_slip_angle, input_mu):
    load_kn = vertical_load_n / 1000.0
    slip_range = np.linspace(0, 100, 500) # 0% to 100% longitudinal slip
    
    # Calculating forces
    fx_values = [calculate_fx(load_kn, s, Fx_coeffs) for s in slip_range]
    fy_value = calculate_fy(load_kn, input_slip_angle, SideForce_coeffs_all, input_mu)
    fy_values = np.full_like(slip_range, fy_value)
    
    # Creating the Plot
    plt.figure(figsize=(10, 6))
    plt.plot(slip_range, fx_values, label='Brake Force (Fx)')
    plt.plot(slip_range, fy_values, label=f'Side Force (Fy) at {input_slip_angle}°')
    plt.xlabel('Longitudinal Slip [%]')
    plt.ylabel('Force [N]')
    plt.title(f'Tyre Forces (Weight: {args.weight}kg, mu: {input_mu})')
    plt.legend()
    plt.grid(True)
    
    # Saving the file
    save_name = f"plot_{args.slip}_{args.weight}_{input_mu}.png"
    plt.savefig(save_name)
    print(f"Plot saved as {save_name}")

# we run the code and generate/save the plot as png
final_plot(fz, args.slip, args.mu)