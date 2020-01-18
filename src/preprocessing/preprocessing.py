import logging

from numba import njit
import numpy as np
import pandas as pd
from pyjet import cluster, DTYPE_PTEPM

logging.basicConfig(level=logging.INFO)


@njit('f8[:](f8[:])')
def et_miss(event):
    pts = event[::3]
    phis = event[2::3]
    px = np.sum(pts * np.cos(phis))
    py = np.sum(pts * np.sin(phis))
    # $E_{T,miss}$ is defined as minus the sum of the transverse momenta
    return -1 * np.array([px, py])


class ClusterJets():
    def __init__(self, has_labels):
        self.alljets = dict()
        self.etmisses = dict()
        self.has_labels = has_labels
        if self.has_labels:
            self.alljets['background'] = list()
            self.alljets['signal'] = list()
            self.etmisses['background'] = list()
            self.etmisses['signal'] = list()
        else:
            self.alljets['unlabeled'] = list()
            self.etmisses['unlabeled'] = list()

    def get_jet_inputs(self, event):
        """
        Process the raw data into a format suitable for feeding into pyjet
        """
        n_consts = np.sum(event[::3] > 0)
        inputs = event[:3 * n_consts].reshape(n_consts, 3)
        massless = np.zeros(n_consts).reshape(-1, 1)
        fourvecs = np.append(inputs, massless, axis=1)
        return np.rec.fromarrays(fourvecs.T, dtype=DTYPE_PTEPM)

    def cluster_jet(self, event, target=None, jet_algo=-1, jet_size=1.0, pt_cut=20):
        """
        Cluster an event into jets and hold them in `alljets`
        """
        pseudojets_input = self.get_jet_inputs(event)
        # jet_algos: -1 = anti-kt, 1 = kt, 0 = cambridge-aachen
        sequence = cluster(pseudojets_input, R=jet_size, p=jet_algo)
        jets = sequence.inclusive_jets(ptmin=pt_cut)
        # while we're here grab the missing transverse energy
        missing_et = et_miss(event)
        # append event to the appropriate list
        if target is None:
            self.alljets['unlabeled'].append(jets)
            self.etmisses['unlabeled'].append(missing_et)
        elif event[target] == 0:
            self.alljets['background'].append(jets)
            self.etmisses['background'].append(missing_et)
        else:
            self.alljets['signal'].append(jets)
            self.etmisses['signal'].append(missing_et)

    def cluster_jets(self, events):
        """
        Cluster all the raw data into jets and hold them in `alljets`
        """
        # when labeled, the last column of the event is the target
        if self.has_labels:
            target_column = events.columns[-1]
        else:
            target_column = None
        for event in events.values:
            self.cluster_jet(event, target_column)


def four_vector(jet):
    return np.array([jet.pt, jet.eta, jet.phi, jet.mass])

@njit('f8(f8, f8)')
def delta_phi(phi_1, phi_2):
    """
    $\Delta\phi$ takes values between 0 and $\pi$ whereas $\phi$ takes values between $-\pi$ and $\pi$
    """
    return min(abs(phi_1 - phi_2), 2 * np.pi - abs(phi_1 - phi_2))

@njit('f8(f8, f8, f8, f8)')
def delta_R(eta_1, eta_2, phi_1, phi_2):
    delta_eta = eta_1 - eta_2
    return (delta_eta**2 + delta_phi(phi_1, phi_2)**2)**0.5

@njit('f8[:](f8[:], f8[:,:])')
def subjets_delta_R(jet_four_vector, jet_constituents_array):
    jet_eta = jet_four_vector[1]
    jet_phi = jet_four_vector[2]
    drs = []
    for i in range(len(jet_constituents_array)):
        subjet = jet_constituents_array[i]
        drs.append(delta_R(jet_eta, subjet[1], jet_phi, subjet[2]))
    return np.array(drs)

def get_n_jets(ktjets, n):
    return np.array([four_vector(njet) for njet in ktjets.exclusive_jets(n)])

@njit('f8[:,:](f8[:,:], f8[:,:])')
def subjets_ktjets_delta_R(njet_four_vectors, jet_constituents_array):
    drs0 = []
    for j in range(len(njet_four_vectors)):
        njet = njet_four_vectors[j]
        njet_eta = njet[1]
        njet_phi = njet[2]
        drs1 = []
        for i in range(len(jet_constituents_array)):
            subjet = jet_constituents_array[i]
            drs1.append(delta_R(njet_eta, subjet[1], njet_phi, subjet[2]))
        drs0.append(drs1)
    return np.array(drs0)

def get_n_subjetiness(pts, ktjets, subjet_array, n):
    """
    n-subjetiness, $\tau_n^{(\beta)}$, is definied in arXiv:1011.2268
    """
    njets = get_n_jets(ktjets, n)
    drs = np.min(subjets_ktjets_delta_R(njets, subjet_array).T, axis=1)
    return np.sum(pts * drs)

def get_substructure(jet, jet_size=1.0):
    # I will eventually drop NaNs, so get that out of the way first
    ktjets = cluster(jet, R=jet_size, p=1)
    if len(ktjets.inclusive_jets()[0].constituents()) < 4:
        return np.array([np.nan] * 6)

    # vector of p_T for each constituent
    pts = jet.constituents_array()['pT']
    # multiplicity
    multiplicity = len(pts)
    # normalization factor for many substructure observables
    norm = np.sum(pts)
    # fragmentation function
    fragmentation = np.sqrt(np.sum(pts**2)) / norm

    # vector of \delta R between the jet and each constituent
    subjet_array = np.array([four_vector(subjet) for subjet in jet.constituents()])
    subjets_dr = subjets_delta_R(four_vector(jet), subjet_array)
    # the numerator of \tau_1^{(1)}
    one = np.sum(pts * subjets_dr)
    # the numerator of \tau_1^{(2)}
    one_b2 = np.sum(pts * subjets_dr**2)

    # look at substructure with exclusive jets
    # the numerator of \tau_2^{(1)}
    two = get_n_subjetiness(pts, ktjets, subjet_array, 2)
    # the numerator of \tau_3^{(1)}
    three = get_n_subjetiness(pts, ktjets, subjet_array, 3)
    # the numerator of \tau_4^{(1)}
    four = get_n_subjetiness(pts, ktjets, subjet_array, 4)

    # return substructure observables
    return np.array([np.sqrt(one_b2 * norm) / one, two / one, three / two, four / three, multiplicity, fragmentation])


class ProcessData():
    def __init__(self):
        self.column_names = (
            'pt_1',
            'eta_1',
            'phi_1',
            'mass_1',
            'pt_2',
            'eta_2',
            'phi_2',
            'mass_2',
            'sqrt(tau1(2))/tau1(1)_1',
            'tau21_1',
            'tau32_1',
            'tau43_1',
            'multiplicity_1',
            'fragmentation_1',
            'sqrt(tau1(2))/tau1(1)_2',
            'tau21_2',
            'tau32_2',
            'tau43_2',
            'multiplicity_2',
            'fragmentation_2'
        )

    def get_processed_row(self, event):
        return np.array([
            event[0].pt,
            event[0].eta,
            event[0].phi,
            event[0].mass,
            event[1].pt,
            event[1].eta,
            event[1].phi,
            event[1].mass,
            *get_substructure(event[0]),
            *get_substructure(event[1])
        ])

    def create_df(self, event_store, key):
        events = event_store.alljets[key]
        processed_events = np.array([self.get_processed_row(event) for event in events])
        missing_ets = np.array(event_store.etmisses[key])

        df1 = pd.DataFrame(data=processed_events, columns=self.column_names)
        df2 = pd.DataFrame(data=missing_ets, columns=['etmiss_x', 'etmiss_y'])
        return pd.merge(df1, df2, how='inner', left_index=True, right_index=True)
