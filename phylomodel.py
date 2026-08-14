import gzip
import os
import sys

import polars as pl

import jax
import jax.numpy as jnp


NStates:int = 18

@jax.jit
def cooccurrence(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    i = a.astype(jnp.int32) - 1
    j = b.astype(jnp.int32) - 1
    flat_indices = i * 18 + j
    counts = jnp.bincount(flat_indices, length=NStates * NStates)
    return counts.reshape((NStates, NStates))

@jax.jit
def logdet(a: jnp.ndarray, b: jnp.ndarray) -> jnp.float64:
    F = cooccurrence(a, b) 
    F = F.astype(jnp.float64) / jnp.sum(F)
    
    row_sum = jnp.sum(F, axis=1)
    col_sum = jnp.sum(F, axis=0)
    
    sign_F, logdet_F = jnp.linalg.slogdet(F)
    logdet_D1 = jnp.sum(jnp.log(row_sum))
    logdet_D2 = jnp.sum(jnp.log(col_sum))
    
    d = -(logdet_F - 0.5 * (logdet_D1 + logdet_D2)) / NStates
    return d

class PhyloStates:
    def __init__(self, datadir):
        self.datafile = f"{datadir}/leaves_data_matrix.csv.gz"
        self.headerfile = f"{datadir}/header"
        self.annotationfile = f"{datadir}/states.txt"
        with gzip.open(self.datafile, 'rt', encoding='utf-8') as f:
            self.data = [line.strip() for line in f]
        with open(self.headerfile) as infile:
            self.header = infile.read()
        with open(self.annotationfile) as infile:
            self.annotation = infile.read()
    def frame(self):
        head = list(self.header.splitlines())
        n_cols = len(head)
        schema = dict([(f"column_{i+1}", pl.Int32 if i == 0 else pl.Int8) for i, _ in enumerate(head)])
        df = pl.read_csv(self.datafile, has_header=False, schema_overrides=schema)
        return df.to_numpy()




def main(args):
    pass

if __name__ == "__main__":
    main(sys.argv[1:])








 





        




