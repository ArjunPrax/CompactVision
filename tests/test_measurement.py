"""Arithmetic/contract tests only; fixtures are not model/service performance."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmarks.report import distribution, load_json, load_jsonl, summarize
from benchmarks.timing import Trace

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/synthetic"
SECOND = 1_000_000_000


class Clock:
    def __init__(self):
        self.now = 100

    def __call__(self):
        return self.now


class TimingTests(unittest.TestCase):
    def test_stage_intervals_gaps_and_repeated_names(self):
        clock = Clock()
        trace = Trace("image", clock=clock)
        clock.now = 110
        with trace.stage("decode"):
            clock.now = 120
        clock.now = 125
        with trace.stage("decode"):
            clock.now = 128
        clock.now = 140
        self.assertEqual(trace.finish(), {
            "request_id": "image", "duration_ns": 40,
            "stages": [
                {"name": "decode", "start_ns": 10, "end_ns": 20},
                {"name": "decode", "start_ns": 25, "end_ns": 28},
            ],
        })

    def test_exception_is_propagated_and_failed_stage_kept(self):
        clock = Clock()
        trace = Trace("bad-image", clock=clock)
        with self.assertRaisesRegex(RuntimeError, "decoder failed"):
            with trace.stage("decode"):
                clock.now += 7
                raise RuntimeError("decoder failed")
        self.assertEqual(trace.finish()["stages"][0]["end_ns"], 7)

    def test_nested_stage_is_rejected_without_corrupting_outer_stage(self):
        trace = Trace("image", clock=Clock())
        with trace.stage("outer"):
            with self.assertRaises(ValueError):
                with trace.stage("inner"):
                    self.fail("nested stages must not execute")
            with self.assertRaises(ValueError):
                trace.finish()
        self.assertEqual(len(trace.finish()["stages"]), 1)

    def test_finished_trace_cannot_be_reused(self):
        trace = Trace("image", clock=Clock())
        trace.finish()
        with self.assertRaises(ValueError):
            trace.finish()
        with self.assertRaises(ValueError):
            with trace.stage("late"):
                self.fail("closed stage executed")

    def test_backwards_clock_fails(self):
        clock = Clock()
        trace = Trace("image", clock=clock)
        clock.now = 99
        with self.assertRaisesRegex(ValueError, "backwards"):
            trace.finish()

    def test_stage_collection_can_be_disabled_without_changing_call_execution(self):
        clock = Clock()
        trace = Trace("image", clock=clock, collect_stages=False)
        with trace.stage("model"):
            clock.now += 13
        self.assertEqual(trace.finish(), {"request_id": "image", "duration_ns": 13, "stages": []})


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_json(FIXTURE / "manifest.json")
        self.requests = load_jsonl(FIXTURE / "requests.jsonl")
        self.traces = load_jsonl(FIXTURE / "traces.jsonl")

    def report(self):
        return summarize(self.manifest, self.requests, self.traces)

    def test_hand_calculated_outcomes_latency_and_window_boundaries(self):
        report = self.report()
        self.assertEqual(report["counts"], {
            "offered": 7, "sent": 6, "success": 2, "rejected": 1,
            "error": 1, "timeout": 1, "outstanding": 1, "generator_drop": 1,
            "success_completed_in_window": 1,
        })
        self.assertEqual(report["client_elapsed_by_outcome_ms"]["success"], {
            "count": 2, "p50_ms": 1500, "p95_ms": 2000,
        })
        self.assertEqual(report["client_elapsed_by_outcome_ms"]["timeout"]["p50_ms"], 5000)
        self.assertEqual(report["successful_completions_per_second_in_window"], 0.1)
        self.assertEqual(report["offered_requests_per_second"], 0.7)
        server = report["server_traces"]["by_client_outcome"]["success"]
        self.assertEqual(server["unattributed_ms"]["p50_ms"], 2)
        self.assertEqual(server["stages_ms"]["model"]["p50_ms"], 4)
        self.assertEqual(report["evidence_kind"], "synthetic")
        self.assertIn("Synthetic fixture", report["warnings"][0])

    def test_late_drain_success_is_not_window_throughput(self):
        self.requests[0]["finished_ns"] = 10 * SECOND
        self.requests[0]["sent_ns"] = 9 * SECOND
        self.assertEqual(self.report()["counts"]["success_completed_in_window"], 0)

    def test_missing_scheduled_arrival_and_duplicate_id_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "row count"):
            summarize(self.manifest, self.requests[:-1])
        self.requests[1]["request_id"] = "a"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.report()

    def test_after_deadline_response_must_not_be_relabelled_success(self):
        self.requests[0]["finished_ns"] = 6 * SECOND
        with self.assertRaisesRegex(ValueError, "remain a timeout"):
            self.report()

    def test_outstanding_past_deadline_and_early_timeout_are_invalid(self):
        self.requests[-1]["sent_ns"] = 8 * SECOND
        self.manifest["observation_end_ns"] = 13 * SECOND
        with self.assertRaisesRegex(ValueError, "not yet timed out"):
            self.report()
        self.setUp()
        self.requests[2]["finished_ns"] = 2 * SECOND
        with self.assertRaisesRegex(ValueError, "before its deadline"):
            self.report()

    def test_empty_success_sample_is_null_not_zero(self):
        for row in self.requests[:2]:
            row["outcome"] = "error"
        self.assertEqual(self.report()["client_elapsed_by_outcome_ms"]["success"], {
            "count": 0, "p50_ms": None, "p95_ms": None,
        })

    def test_malformed_event_timestamps_are_rejected(self):
        for field, value in (("sent_ns", True), ("scheduled_ns", -1), ("finished_ns", 0.5)):
            rows = copy.deepcopy(self.requests)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(self.manifest, rows)

    def test_scheduled_at_end_or_sent_before_schedule_is_invalid(self):
        self.requests[0]["scheduled_ns"] = 10 * SECOND
        with self.assertRaisesRegex(ValueError, "scheduled arrival"):
            self.report()
        self.setUp()
        self.requests[1]["sent_ns"] = 8 * SECOND
        with self.assertRaisesRegex(ValueError, "send must follow"):
            self.report()

    def test_generator_drop_cannot_have_a_send_or_server_trace(self):
        self.requests[5]["sent_ns"] = 4 * SECOND
        with self.assertRaisesRegex(ValueError, "unsent"):
            self.report()
        self.setUp()
        self.traces[0]["request_id"] = "f"
        with self.assertRaisesRegex(ValueError, "cannot have a server trace"):
            self.report()

    def test_unknown_and_duplicate_traces_are_rejected(self):
        self.traces[0]["request_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unique scheduled"):
            self.report()
        self.setUp()
        self.traces.append(copy.deepcopy(self.traces[0]))
        with self.assertRaisesRegex(ValueError, "unique scheduled"):
            self.report()

    def test_overlaps_and_stages_outside_trace_are_rejected(self):
        for start, end in ((1_000_000, 7_000_000), (3_000_000, 9_000_000)):
            traces = copy.deepcopy(self.traces)
            traces[0]["stages"][1].update(start_ns=start, end_ns=end)
            with self.subTest(start=start), self.assertRaisesRegex(ValueError, "nonoverlapping"):
                summarize(self.manifest, self.requests, traces)

    def test_repeated_stage_names_are_summed_per_request_not_extra_samples(self):
        self.traces[0]["stages"] = [
            {"name": "decode", "start_ns": 0, "end_ns": 1_000_000},
            {"name": "decode", "start_ns": 2_000_000, "end_ns": 3_000_000},
        ]
        stage = self.report()["server_traces"]["by_client_outcome"]["success"]["stages_ms"]["decode"]
        self.assertEqual(stage, {"count": 1, "p50_ms": 2, "p95_ms": 2})

    def test_measured_label_requires_metadata(self):
        self.manifest["evidence_kind"] = "measured"
        with self.assertRaisesRegex(ValueError, "missing reproducibility"):
            self.report()

    def test_complete_declared_metadata_is_required_but_not_a_quality_certificate(self):
        self.manifest["evidence_kind"] = "measured"
        metadata = {
            "model_sha256": "a" * 64, "inputs_sha256": "b" * 64,
            "configuration": {
                "batch_size": 1, "service_workers": 1,
                "runtime_threads": {"intra_op": 1}, "other_thread_pools": {}, "runtime_options": {},
            },
            "environment": {
                "hardware": "DECLARED TEST VALUE", "os": "DECLARED TEST VALUE",
                "architecture": "DECLARED TEST VALUE", "python": "DECLARED TEST VALUE",
                "runtime": "DECLARED TEST VALUE", "provider": "DECLARED TEST VALUE",
            },
            "code_revision": "DECLARED TEST VALUE", "warmup_calls": 50,
            "offered_rate_rps": 0.7, "max_in_flight": 10, "retry_policy": "disabled",
            "timing_boundaries": {"client": "DECLARED TEST VALUE", "server": "DECLARED TEST VALUE"},
        }
        self.manifest["reproducibility"] = metadata
        report = self.report()
        self.assertEqual(report["reproducibility_missing"], [])
        self.assertEqual(len(report["configuration_sha256"]), 64)
        reordered = dict(reversed(list(metadata["configuration"].items())))
        metadata["configuration"] = reordered
        self.assertEqual(self.report()["configuration_sha256"], report["configuration_sha256"])
        del metadata["environment"]["provider"]
        with self.assertRaisesRegex(ValueError, "environment.provider"):
            self.report()

    def test_deadline_equality_is_timeout_and_zero_arrivals_keep_null_latency(self):
        self.requests[0]["finished_ns"] = 5 * SECOND
        with self.assertRaisesRegex(ValueError, "remain a timeout"):
            self.report()
        self.manifest["scheduled_requests"] = 0
        report = summarize(self.manifest, [])
        self.assertEqual(report["counts"]["offered"], 0)
        self.assertIsNone(report["client_elapsed_by_outcome_ms"]["success"]["p95_ms"])

    def test_invalid_metadata_and_nonfinite_json_are_rejected(self):
        for metadata in ({"model_sha256": "placeholder"}, {"offered_rate_rps": float("nan")}, {"retry_policy": "on"}):
            self.manifest["reproducibility"] = metadata
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                self.report()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text('{"value": NaN}')
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                load_json(path)

    def test_quantile_convention_on_independent_hand_example(self):
        sample = [1_000_000] * 9 + [20_000_000]
        self.assertEqual(distribution(sample), {"count": 10, "p50_ms": 1, "p95_ms": 20})

    def test_cli_preserves_synthetic_label_and_rejects_bad_files(self):
        command = [sys.executable, "-m", "benchmarks.report", str(FIXTURE / "manifest.json"), str(FIXTURE / "requests.jsonl")]
        good = subprocess.run(command + ["--traces", str(FIXTURE / "traces.jsonl")], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual(json.loads(good.stdout)["evidence_kind"], "synthetic")
        bad = subprocess.run(command[:-1] + [str(FIXTURE / "missing.jsonl")], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn("invalid run", bad.stderr)


if __name__ == "__main__":
    unittest.main()
