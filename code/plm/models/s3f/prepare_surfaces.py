#!/usr/bin/env python3
"""Utilities for memory-bounded S3F surface preprocessing."""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np


CURVATURE_CHUNK_SIZE = 512


def load_s3f_module(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_surface_precompute_s3f", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load S3F script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def chunked_compute_curvatures(surface, xyz, normals, batch, curvature_scales):
    """Compute the official S3F curvature formula in query-point chunks."""
    import torch
    from torch.nn import functional as F

    n_points = xyz.shape[0]
    scales = xyz.new_tensor(curvature_scales)
    normals_by_scale = torch.empty(
        (n_points, len(scales), 3), dtype=xyz.dtype, device=xyz.device
    )
    for start in range(0, n_points, CURVATURE_CHUNK_SIZE):
        end = min(start + CURVATURE_CHUNK_SIZE, n_points)
        distance2 = torch.cdist(xyz[start:end], xyz).square()
        kernel = torch.exp(
            -distance2[..., None] / (2 * scales[None, None, :].square())
        )
        normals_by_scale[start:end] = F.normalize(
            torch.einsum("ijs,jd->isd", kernel, normals), p=2, dim=-1
        )

    tangents_by_scale = surface.tangent_vectors(normals_by_scale)
    features = []
    for scale_index, scale in enumerate(scales):
        scale_normals = normals_by_scale[:, scale_index, :].contiguous()
        scale_tangents = tangents_by_scale[:, scale_index, :, :].contiguous()
        mean_curvature = torch.empty(n_points, dtype=xyz.dtype, device=xyz.device)
        gauss_curvature = torch.empty_like(mean_curvature)
        for start in range(0, n_points, CURVATURE_CHUNK_SIZE):
            end = min(start + CURVATURE_CHUNK_SIZE, n_points)
            diff_positions = xyz[None, :, :] - xyz[start:end, None, :]
            diff_normals = (
                scale_normals[None, :, :] - scale_normals[start:end, None, :]
            )
            normal_dot = (
                scale_normals[start:end, None, :] * scale_normals[None, :, :]
            ).sum(dim=-1)
            distance2 = diff_positions.square().sum(dim=-1) * (2 - normal_dot).square()
            window = torch.exp(-distance2 / (2 * scale.square()))
            tangents = scale_tangents[start:end]
            projected_positions = torch.einsum(
                "iab,ijb->ija", tangents, diff_positions
            )
            projected_normals = torch.einsum(
                "iab,ijb->ija", tangents, diff_normals
            )
            projected = torch.cat([projected_positions, projected_normals], dim=-1)
            covariance = (
                window[:, :, None, None]
                * projected_positions[:, :, :, None]
                * projected[:, :, None, :]
            ).sum(dim=1)
            covariance = covariance.view(end - start, 2, 2, 2)
            position_covariance = covariance[:, :, 0, :]
            normal_covariance = covariance[:, :, 1, :]
            position_covariance[:, 0, 0] += 1e-10
            position_covariance[:, 1, 1] += 1e-10
            normal_covariance[:, 0, 0] += 1e-10
            normal_covariance[:, 1, 1] += 1e-10
            shape = torch.linalg.lstsq(
                normal_covariance, position_covariance
            ).solution
            a, b = shape[:, 0, 0], shape[:, 0, 1]
            c, d = shape[:, 1, 0], shape[:, 1, 1]
            mean_curvature[start:end] = (a + d).clamp(-1, 1)
            gauss_curvature[start:end] = (a * d - b * c).clamp(-1, 1)
        features.extend([mean_curvature, gauss_curvature])

    result = torch.stack(features, dim=-1)
    result[torch.isnan(result)] = 0
    return result


def install_chunked_curvature() -> None:
    from s3f import surface

    surface.compute_curvatures = lambda xyz, normals, batch, curvature_scales: (
        chunked_compute_curvatures(surface, xyz, normals, batch, curvature_scales)
    )


def validate_surface(path: Path, sequence_length: int) -> dict[str, int]:
    with path.open("rb") as handle:
        surface = pickle.load(handle)
    required = {
        "surf_points",
        "surf_normals",
        "surf_hks",
        "surf_curvatures",
        "res2surf",
    }
    if not required.issubset(surface):
        raise ValueError(f"{path}: incomplete S3F surface graph")
    points = np.asarray(surface["surf_points"])
    normals = np.asarray(surface["surf_normals"])
    res2surf = np.asarray(surface["res2surf"])
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError(f"{path}: invalid surface coordinates")
    if res2surf.ndim != 3 or res2surf.shape[0] != sequence_length:
        raise ValueError(f"{path}: res2surf does not match the WT sequence")
    if points.shape[0] == 0 or res2surf.min() < 0 or res2surf.max() >= points.shape[0]:
        raise ValueError(f"{path}: invalid residue-to-surface mapping")
    for key in ("surf_points", "surf_normals", "surf_hks", "surf_curvatures"):
        values = np.asarray(surface[key])
        if values.shape[0] != points.shape[0] or not np.isfinite(values).all():
            raise ValueError(f"{path}: invalid {key}")
    return {
        "surface_points": int(points.shape[0]),
        "res2surf_width": int(np.prod(res2surf.shape[1:])),
    }
