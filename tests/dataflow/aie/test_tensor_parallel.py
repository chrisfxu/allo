# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo.dataflow as df
from allo.ir.types import int16, int32
import numpy as np
import allo

def _test_tensor_parallelism():
    Ty = int32
    M, K, N, L = 256, 256, 256, 256

    P = 4
    Nt = N // P

    @df.region()
    def top():
        pipe = df.array(df.pipe(dtype=Ty, shape=(M, L), depth=2), shape=(P,))

        @df.kernel(mapping=[P])
        def mul(X: Ty[M, K], W1: Ty[K, N], W2: Ty[N, L], Z: Ty[M, L]):
            pi = df.get_pid()
            Y_out: Ty[M, Nt] = allo.matmul(
                X, W1[:, pi * Nt : (pi + 1) * Nt]
            )
            Z_out: Ty[M, L] = allo.matmul(
                Y_out, W2[pi * Nt : (pi + 1) * Nt, :]
            )
            pipe[pi].put(Z_out)

        def acc(X: Ty[M, K], W1: Ty[K, N], W2: Ty[N, L], Z: Ty[M, L]):
            Z_out1 = pipe[0].get()
            Z_out2 = pipe[1].get()
            Z_out3 = pipe[2].get()
            Z_out4 = pipe[3].get()
            for i in range(M):
                for j in range(L):
                    Z[i, j] += Z_out1[i, j]
                    Z[i, j] += Z_out2[i, j]
                    Z[i, j] += Z_out3[i, j]
                    Z[i, j] += Z_out4[i, j]

    X = np.random.randint(0, 64, (M, K)).astype(np.int32)
    W1 = np.random.randint(0, 64, (K, N)).astype(np.int32)
    W2 = np.random.randint(0, 64, (N, L)).astype(np.int32)

    # if "MLIR_AIE_INSTALL_DIR" in os.environ:
    #     mod = df.build(top, target="aie")
    #     Z = np.zeros((M, L)).astype(np.int32)
    #     mod(X, W1, W2, Z)
    #     np.testing.assert_allclose(Z, X @ W1 @ W2, atol=1e-5)
    #     print("PASSED!")
    # else:
    #     print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")

    # AIE broken right now
    sim_mod = df.build(top, target="simulator")
    Z = np.zeros((M, L)).astype(np.int32)
    sim_mod(X, W1, W2, Z)
    np.testing.assert_allclose(Z, X @ W1 @ W2, atol=1e-5)
    print("Dataflow Simulator Passed!")

if __name__ == "__main__":
   _test_tensor_parallelism()
