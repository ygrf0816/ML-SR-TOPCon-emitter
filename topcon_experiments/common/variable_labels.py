"""Human-readable labels for topcon_dataset.csv columns."""

from __future__ import annotations

COLUMN_LABELS: dict[str, str] = {
    "file_base": "sample_id (file_base)",
    "athena_thick": "BSG oxide thickness (um) [athena_thick]",
    "athena_c_boron": "Boron concentration in deposit (cm-3) [athena_c_boron]",
    "athena_temp1": "1st diffusion temperature (C) [athena_temp1]",
    "athena_time1": "1st diffusion time (min) [athena_time1]",
    "athena_temp2": "2nd diffusion temperature (C) [athena_temp2]",
    "athena_time2": "2nd diffusion time (min) [athena_time2]",
    "athena_F_N2": "N2 flow coefficient [athena_F_N2]",
    "athena_F_O2": "O2 flow coefficient [athena_F_O2]",
    "doping_N_peak": "Doping peak concentration (cm-3) [doping_N_peak]",
    "doping_x_peak": "Doping peak depth (um) [doping_x_peak]",
    "doping_junction_depth": "Doping junction depth (um) [doping_junction_depth]",
    "doping_FWHM": "Doping FWHM (um) [doping_FWHM]",
    "doping_gradient_max": "Doping max gradient (cm-3/um) [doping_gradient_max]",
    "doping_dose": "Doping dose integral (cm-3*um) [doping_dose]",
    "doping_R_sheet": "Front emitter sheet resistance (ohm/sq) [doping_R_sheet]",
    "defect_vac_N_peak": "Vacancy peak (cm-3) [defect_vac_N_peak]",
    "defect_vac_gradient_max": "Vacancy max gradient [defect_vac_gradient_max]",
    "defect_vac_dose": "Vacancy dose integral [defect_vac_dose]",
    "iv_Voc": "Open-circuit voltage (V) [iv_Voc]",
    "iv_Jsc": "Short-circuit current density (mA/cm2) [iv_Jsc]",
    "iv_FF": "Fill factor (%) [iv_FF]",
    "iv_Eff": "Power density / PCE proxy (mW/cm2) [iv_Eff]",
}

# Targets where absolute RMSE/MAE are misleading due to magnitude
LARGE_MAGNITUDE_TARGETS = {
    "doping_N_peak",
    "doping_dose",
    "doping_gradient_max",
    "defect_vac_N_peak",
    "defect_vac_gradient_max",
    "defect_vac_dose",
}


def label_for(col: str) -> str:
    return COLUMN_LABELS.get(col, col)
