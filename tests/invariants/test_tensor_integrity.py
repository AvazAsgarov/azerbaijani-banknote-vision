"""Pipeline Invariant Tests for Vision Transformer Latent Tensor Integrity.

Enforces strict numerical and dimensional contracts: feature arrays must
maintain exact (N=2522, D=1024) shapes, unit L2 norms, and zero NaN/Inf values.
"""

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import pytest

from src.core.config import ProjectPaths


@pytest.mark.invariant
class TestTensorIntegrityInvariants:
    """Mathematical invariants verifying numerical stability of latent embeddings."""

    @pytest.fixture(scope="class")
    def paths(self) -> ProjectPaths:
        """Returns project paths container."""
        return ProjectPaths()

    def test_hdf5_embeddings_contract(self, paths: ProjectPaths) -> None:
        """Verifies shape, normalization, and absence of NaN/Inf in HDF5 archive."""
        assert paths.embeddings_h5_path.exists(), f"Missing H5 archive: {paths.embeddings_h5_path}"

        with h5py.File(paths.embeddings_h5_path, "r") as h5f:
            assert "embeddings" in h5f, "Dataset 'embeddings' missing from H5 container"
            assert "filenames" in h5f, "Dataset 'filenames' missing from H5 container"

            embs = h5f["embeddings"][:]
            fnames = h5f["filenames"][:]

            # Dimensional shape assertion: exactly 2522 samples, 1024 latent dimensions
            assert embs.shape == (2522, 1024), f"Unexpected embedding shape: {embs.shape}"
            assert len(fnames) == 2522, f"Unexpected filename count: {len(fnames)}"

            # Numerical stability assertion: zero NaN or Inf values
            assert not np.isnan(embs).any(), "NaN values detected in latent embedding tensor!"
            assert not np.isinf(embs).any(), "Infinite values detected in latent embedding tensor!"

            # Unit L2-norm normalization assertion
            norms = np.linalg.norm(embs, axis=1)
            np.testing.assert_allclose(norms, np.ones(2522), atol=1e-3, err_msg="Non-unit L2 norms discovered!")

    def test_numpy_and_parquet_synchronization(self, paths: ProjectPaths) -> None:
        """Verifies parity between HDF5, NumPy, and Parquet metadata stores."""
        assert paths.embeddings_npy_path.exists(), f"Missing NPY array: {paths.embeddings_npy_path}"
        assert paths.embeddings_parquet_path.exists(), f"Missing Parquet store: {paths.embeddings_parquet_path}"

        npy_embs = np.load(paths.embeddings_npy_path)
        assert npy_embs.shape == (2522, 1024)

        with h5py.File(paths.embeddings_h5_path, "r") as h5f:
            h5_embs = h5f["embeddings"][:]
            np.testing.assert_allclose(npy_embs, h5_embs, atol=1e-5)

        df_meta = pd.read_parquet(paths.embeddings_parquet_path)
        assert len(df_meta) == 2522
        assert not df_meta["split"].isna().any()
