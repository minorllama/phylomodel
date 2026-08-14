# match original numpy implementation against jax/polars
# disable nengo pytest plugin if installed `pytest -p no:nengo` 

import random
import pytest


import numpy as np

from phylomodel import *
from collections import Counter

@pytest.fixture(scope="module")
def phylo_instance():
    """instantiate PhyloStates once for the test module."""
    return PhyloStates("./data")


@pytest.fixture(scope="module")
def phylo_data(phylo_instance):
    """load the data frame once for the test module."""
    return phylo_instance.frame()

def np_parser(cfg, k=-1):
    N = len(cfg.data)
    columns = [e for e in cfg.header.splitlines()] 
    selector = lambda es, k: es[1:] if k <= 1 else es[1:k+1]
    samples = selector(columns, k)
    assert columns[0] == "bin_start"
    vectorized = [np.zeros(N, dtype=np.int8) for sample in samples]
    for i, e in enumerate(cfg.data):
        state_data = e.split(",")
        start = state_data[0]
        states = selector(state_data, k)
        assert len(states) == len(samples), [i, len(states), len(samples), states, samples]
        for sample_k, state in enumerate(states):
            vectorized[sample_k][i] = np.int8(state)
    return vectorized                

def np_cooccurrence(s1, s2, N):
    J12 = np.zeros((N, N))
    assert len(s1) == len(s2)
    for k, v in Counter(zip(s1, s2)).items():
        x, y = k[0]-1, k[1]-1
        J12[x][y] = v
    return J12

def np_logdet(s1, s2, N):
    J12 = np_cooccurrence(s1, s2, N)
    J12 = J12/np.sum(J12)
    D1 = np.diag(J12.sum(axis=1))
    D2 = np.diag(J12.sum(axis=0))
    sd1, d1 = np.linalg.slogdet(D1)
    sd2, d2 = np.linalg.slogdet(D2)
    sJ, J = np.linalg.slogdet(J12)
    if not ( d1 != 0 and d2 != 0 and J != 0):
        raise Exception([tag, [d1, d2, J,  J12]])
    d =  -( J - d1/2 - d2/2) #-np.log(J / np.sqrt(d1*d2)) ## this has overflow/underflow for large sequences
    return d/N

def test_frame(phylo_data, phylo_instance):
    k = 100
    data_selected = phylo_data[:, 1:k+1]
    data_np = np.array(np_parser(phylo_instance, k)).transpose()
    err = np.linalg.norm(data_selected - data_np)
    assert err == 0

@pytest.mark.parametrize("n_states", [NStates])
@pytest.mark.parametrize("K", [10])
def test_logdet(phylo_data, n_states, K, atol = 0.001):
    data = phylo_data
    (n_bins, n_samples) = data.shape
    for k in range(K):
        s1 = random.randint(0, n_samples-1)
        s2 = random.randint(0, n_samples-1)
        d1 = np_logdet(data[:, s1], data[:, s2], n_states)
        d2 = logdet(data[:, s1], data[:, s2])
        assert np.isclose(np.linalg.norm(d2 - d1), 0, atol=atol), [k, d1, d2, s1, s2] 

@pytest.mark.parametrize("n_states", [NStates])
@pytest.mark.parametrize("K", [10])
def test_cooccurennce(phylo_data, n_states, K):
    data = phylo_data
    (n_bins, n_samples) = data.shape
    for k in range(K):
        s1 = random.randint(0, n_samples-1)
        s2 = random.randint(0, n_samples-1)
        m1 = cooccurrence(data[:, s1], data[:, s2])
        m2 = np_cooccurrence(data[:, s1], data[:, s2], n_states)
        assert np.isclose(np.linalg.norm(m2 - m1), 0) 

