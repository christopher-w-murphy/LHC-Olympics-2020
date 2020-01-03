import numpy as np
import pandas as pd
from .utils import get_data


def combine_samples(dataset, keys):
    assert keys[0].split('_')[0] in ('signal', 'background', 'unlabeled')

    df = get_data(dataset, key=keys[0])
    if len(keys) > 1:
        for key in keys[1:]:
            df = df.append(get_data(dataset, key), ignore_index=True)
    df = df.dropna()

    if keys[0].split('_')[0] in ('signal', 'background'):
        t = 1 if keys[0].startswith('signal') else 0
        df['is_signal'] = [t] * len(df)

    df['multiplicity_1'] = df['multiplicity_1'].astype('int64')
    df['multiplicity_2'] = df['multiplicity_2'].astype('int64')

    return df


def combine_signal_background(signal, background):
    return signal.append(background).sample(frac=1).reset_index(drop=True)


class Kinematics():
    def __init__(self):
        pass

    def delta_eta(self, eta_1, eta_2):
        return eta_1 - eta_2

    def delta_phi(self, phi_1, phi_2):
        return np.minimum(np.abs(phi_1 - phi_2), 2 * np.pi - np.abs(phi_1 - phi_2))

    def delta_R(self, eta_1, eta_2, phi_1, phi_2):
        return np.sqrt(self.delta_eta(eta_1, eta_2)**2 + self.delta_phi(phi_1, phi_2)**2)

    def invariant_mass(self, mass_1, mass_2, pt_1, pt_2, eta_1, eta_2, phi_1, phi_2):
        et_1 = np.sqrt(mass_1**2 + pt_1**2)
        et_2 = np.sqrt(mass_2**2 + pt_2**2)
        p1_dot_p2 = et_1 * et_2 * np.cosh(self.delta_eta(eta_1, eta_2)) - pt_1 * pt_2 * np.cos(self.delta_phi(phi_1, phi_2))
        return np.sqrt(mass_1**2 + mass_2**2 + 2 * p1_dot_p2)

    def mass_sum(self, mass_1, mass_2):
        return mass_1 + mass_2

    def mass_diff(self, mass_1, mass_2):
        return np.abs(mass_1 - mass_2)
