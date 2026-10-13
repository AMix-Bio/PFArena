"""CPU-only regression tests for the published benchmark input contract."""
import contextlib
import hashlib
import io
import json
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from common_io import sha256_file, sha256_json
from dataset_io import load_dataset, resolve_dataset_path
from prepare_dataset import TASKS, main, parse_mutant
from prepare_structures import prepare_structures


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.benchmark = self.root / "benchmark"
        self.benchmark.mkdir()
        for chain in ("AC", "DE"):
            (self.benchmark / f"{chain}.a3m").write_text(f">query\n{chain}\n>hit\n{chain}\n")
        for index, task in enumerate(TASKS, 1):
            directory = self.benchmark / task
            directory.mkdir()
            row = dict(assay_id="example", wildtype_sequence="AC:DE", sequence_length=4,
                       DMS_filename="candidates.csv", a3m_relative_path="AC.a3m;DE.a3m")
            mutation = "A1V" if index == 1 else "A1V+D3N"
            sequence = "VC:DE" if index == 1 else "VC:NE"
            pd.DataFrame([dict(candidate_group_id=f"example_T{index}", mutant=mutation,
                               mutated_sequence=sequence, DMS_score=999)]).to_csv(directory / "candidates.csv", index=False)
            if index == 3:
                row.update(anchor_mutant="A1V", anchor_DMS_score=0.5)
            if index == 4:
                row["single_context_data_path"] = f"{task}/context.csv"
                pd.DataFrame([dict(mutant="D3N", mutated_sequence="AC:NE", DMS_score=0.7)]).to_csv(directory / "context.csv", index=False)
            pd.DataFrame([row]).to_csv(directory / "assay.csv", index=False)

    def run_prepare(self, name="canonical"):
        output = self.root / name
        argv = ["prepare_dataset.py", "--benchmark-root", str(self.benchmark), "--output-dir", str(output)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            main()
        return output

    def test_published_layout_and_target_label_isolation(self):
        output = self.run_prepare()
        metadata, proteins, samples, substitutions, queries, contexts, _, msa = load_dataset(output)
        self.assertEqual(metadata["queries"], 4)
        self.assertEqual(queries.query_id.tolist(), ["candidates", "candidates", "example_T3", "example_T4"])
        self.assertEqual(len(contexts), 2)
        self.assertEqual(set(substitutions.chain_id), {1, 2})
        self.assertNotIn("DMS_score", samples.columns)
        self.assertTrue(all(not Path(p).is_absolute() and resolve_dataset_path(output, p).is_file() for p in msa.a3m_path))
        for task in TASKS:
            path = self.benchmark / task / "candidates.csv"
            frame = pd.read_csv(path)
            frame["DMS_score"] = -12345
            frame.to_csv(path, index=False)
        second = self.run_prepare("second")
        self.assertEqual(metadata["dataset_hash"], json.loads((second / "dataset.json").read_text())["dataset_hash"])
        with self.assertRaises(FileExistsError):
            self.run_prepare()

    def test_flat_query_ids_match_shared_evaluator(self):
        with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1]), *sys.path]):
            from evaluation.full import fallback_assay_id
            from router import DATASET_INPUTS
        from prepare_dataset import query_id
        self.assertEqual(TASKS, tuple(DATASET_INPUTS[str(i)][0] for i in range(1, 5)))
        for task, filename in zip(TASKS[:2], ("example.csv", "example_random_multi.csv")):
            path = Path(filename)
            frame = pd.DataFrame({"candidate_group_id": ["example_T1"]})
            self.assertEqual(query_id(task, path, frame, "example"),
                             fallback_assay_id(path, nested_groups=False, setting=task))

    def test_invalid_sequence_and_mutation_fail(self):
        for mutant in ("A1V:A1G", "D1V", "A9V", "A1A"):
            with self.assertRaises(ValueError):
                parse_mutant(mutant, "ACDE")
        path = self.benchmark / TASKS[0] / "candidates.csv"
        frame = pd.read_csv(path)
        frame["mutated_sequence"] = "AC:DE"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "mutated_sequence mismatch"):
            self.run_prepare()

    def test_eve_preparation_uses_explicit_chain_parameters(self):
        output = self.run_prepare()
        theta = self.root / "theta.csv"
        pd.DataFrame([dict(context_sha256=hashlib.sha256(s.encode()).hexdigest(), theta=t)
                      for s, t in (("AC", 0.01), ("DE", 0.2))]).to_csv(theta, index=False)
        destination = self.root / "eve"
        result = subprocess.run([
            sys.executable, str(Path(__file__).parent / "models/s3f_msa/prepare_inputs.py"),
            "--dataset-dir", str(output), "--output-dir", str(destination),
            "--theta-map", str(theta),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        mapping = pd.read_csv(destination / "mapping.csv")
        self.assertEqual(set(mapping.MSA_theta), {0.01, 0.2})
        assignments = pd.read_csv(destination / "sample_mapping.csv")
        self.assertEqual(assignments.sample_id.nunique(), 6)

    def test_finalizer_preserves_public_query_ids(self):
        output = self.run_prepare()
        metadata, _, samples, _, _, contexts, _, _ = load_dataset(output)
        rows = pd.concat([samples, contexts], ignore_index=True)
        predictions = rows[["sample_id", "task", "query_id", "source_assay", "mutant"]].rename(
            columns={"task": "subset", "query_id": "candidate_group_id"})
        predictions["status"] = "ok"
        # Synthetic scores test only the I/O contract, not model inference.
        predictions["masked_marginal"] = 1.0
        predictions["mutant_logp"] = -1.0
        predictions["wt_logp"] = -2.0
        run = self.root / "run"
        (run / "shards").mkdir(parents=True)
        predictions.to_csv(run / "shards/shard_0000.csv", index=False)
        schema = Path(__file__).parent / "models/esm2/output_schema_candidates.json"
        config = {"model_id": "io-test", "output_schema_sha256": sha256_file(schema)}
        (run / "shards/shard_0000.json").write_text(json.dumps(dict(
            dataset_id=metadata["dataset_id"], dataset_hash=metadata["dataset_hash"],
            num_shards=1, shard_id=0, samples=len(predictions), run_config=config,
            run_config_sha256=sha256_json(config))))
        result = subprocess.run([
            sys.executable, str(Path(__file__).parent / "finalize_results.py"),
            "--dataset-dir", str(output), "--run-dir", str(run), "--output-schema", str(schema),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for index, task in enumerate(TASKS, 1):
            table = pd.read_csv(run / "tasks" / task / "evaluator_predictions.csv")
            self.assertEqual(table.assay.tolist(), ["candidates" if index <= 2 else f"example_T{index}"])

    def test_structure_rebinding_validates_actual_pdb(self):
        output = self.run_prepare()
        rows = []
        for sequence, names in (("AC", ["ALA", "CYS"]), ("DE", ["ASP", "GLU"])):
            pdb = self.root / f"{sequence}.pdb"
            pdb.write_text("".join(
                f"ATOM  {i:5d}  CA  {aa} A{i:4d}    {0:8.3f}{0:8.3f}{0:8.3f}{1:6.2f}{90:6.2f}           C\n"
                for i, aa in enumerate(names, 1)))
            rows.append(dict(context_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                             dataset_hash="historical", sequence_length=2, structure_status="ready",
                             structure_qc="review", structure_source="AlphaFold 3",
                             pdb_path=pdb.name, pdb_sha256=sha256_file(pdb)))
        manifest = self.root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            result = prepare_structures(output, manifest)
        registered = pd.read_csv(result)
        self.assertEqual(set(registered.dataset_hash), {json.loads((output / "dataset.json").read_text())["dataset_hash"]})
        self.assertEqual(set(registered.structure_qc), {"review"})
        self.assertTrue(all(not Path(p).is_absolute() for p in registered.pdb_path))
        second = self.run_prepare("second")
        pdb = self.root / "AC.pdb"
        pdb.write_text(pdb.read_text().replace("CYS", "ALA"))
        rows[0]["pdb_sha256"] = sha256_file(pdb)
        pd.DataFrame(rows).to_csv(manifest, index=False)
        with self.assertRaisesRegex(ValueError, "structure sequence"):
            prepare_structures(second, manifest)


if __name__ == "__main__":
    unittest.main()
