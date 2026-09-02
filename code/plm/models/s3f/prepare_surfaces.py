#!/usr/bin/env python3
"""Precompute S3F surface graphs without loading the S3F model."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CURVATURE_CHUNK_SIZE = 512
sys.path.insert(0, str(PROJECT_ROOT))

from common_io import atomic_write_json, sha256_file
from run import balanced_shards, validate_dataset, validate_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--surface-dir", type=Path, required=True)
    parser.add_argument("--s3f-script", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--wt-id")
    return parser.parse_args()


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


def validate_existing(
    output_path: Path,
    metadata_path: Path,
    context,
    pdb_path: Path,
    s3f_script: Path,
) -> dict[str, int]:
    if not output_path.is_file() or not metadata_path.is_file():
        raise FileExistsError(f"incomplete surface cache: {output_path.parent}")
    summary = validate_surface(output_path, int(context["sequence_length"]))
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "wt_id": str(context.name),
        "sequence_sha256": str(context["sequence_sha256"]),
        "sequence_length": int(context["sequence_length"]),
        "pdb_sha256": sha256_file(pdb_path),
        "surface_sha256": sha256_file(output_path),
        "s3f_script_sha256": sha256_file(s3f_script),
        "preprocessor_sha256": sha256_file(Path(__file__)),
        "curvature_implementation": "chunked_pytorch_equivalent_v1",
        "curvature_chunk_size": CURVATURE_CHUNK_SIZE,
        **summary,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"{output_path.parent}: cached surface {field} differs")
    return summary


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must satisfy 0 <= shard-id < num-shards")

    dataset = json.loads((args.dataset_dir / "dataset.json").read_text())
    proteins = pd.read_csv(args.dataset_dir / "proteins.csv")
    validate_dataset(args.dataset_dir, dataset, proteins)
    _, contexts = validate_inputs(args.input_dir, dataset, proteins)
    if args.wt_id:
        if args.wt_id not in set(contexts["wt_id"]):
            raise ValueError(f"unknown wt-id: {args.wt_id}")
        selected = [args.wt_id]
    else:
        selected = balanced_shards(contexts, args.num_shards)[args.shard_id]

    import torch
    from torch.nn import functional as F

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    module = load_s3f_module(args.s3f_script)
    install_chunked_curvature()
    context_by_wt = contexts.set_index("wt_id")
    for wt_id in selected:
        context = context_by_wt.loc[wt_id]
        pdb_path = Path(context["pdb_path"])
        output_dir = args.surface_dir / wt_id
        output_path = output_dir / f"{wt_id}.pkl"
        metadata_path = output_dir / f"{wt_id}.json"
        if output_path.exists() or metadata_path.exists():
            summary = validate_existing(
                output_path, metadata_path, context, pdb_path, args.s3f_script
            )
            print(
                f"{wt_id}: {summary['surface_points']} surface points (reused)",
                flush=True,
            )
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        seed = int(str(context["sequence_sha256"])[:8], 16)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        temporary = output_dir / f".{wt_id}.{os.getpid()}.tmp.pkl"
        try:
            module._write_surface_from_pdb(
                str(pdb_path), str(temporary), torch, F, device
            )
            summary = validate_surface(temporary, int(context["sequence_length"]))
            os.replace(temporary, output_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        atomic_write_json(
            {
                "wt_id": wt_id,
                "sequence_sha256": str(context["sequence_sha256"]),
                "sequence_length": int(context["sequence_length"]),
                "pdb_path": str(pdb_path.resolve()),
                "pdb_sha256": sha256_file(pdb_path),
                "surface_path": str(output_path.resolve()),
                "surface_sha256": sha256_file(output_path),
                "s3f_script_sha256": sha256_file(args.s3f_script),
                "preprocessor_sha256": sha256_file(Path(__file__)),
                "curvature_implementation": "chunked_pytorch_equivalent_v1",
                "curvature_chunk_size": CURVATURE_CHUNK_SIZE,
                "seed": seed,
                "device": str(device),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                **summary,
            },
            metadata_path,
        )
        print(f"{wt_id}: {summary['surface_points']} surface points", flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
