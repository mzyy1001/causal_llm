"""Offline smoke checks. Scripted responses below are not model evaluations."""
import itertools
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class OfflineTests(unittest.TestCase):
    def test_seed_parser(self):
        from run import seeds
        self.assertEqual(seeds("2-4,0,2"), [0, 2, 3, 4])
        with self.assertRaises(ValueError):
            seeds("4-2")

    def test_survival_rate_parser(self):
        from scripts.factorial_agent import parse_rate
        for value in ["88.0%", "0.88", 0.88, 88.0]:
            self.assertAlmostEqual(parse_rate(value), .88)
        self.assertIsNone(parse_rate(None))

    def test_causalab_budget_and_replay(self):
        from causalab.env import CausaLabEnv
        a, b = CausaLabEnv(k=10, seed=1000), CausaLabEnv(k=10, seed=1000)
        self.assertEqual(a.observe(), b.observe())
        a.observe()
        self.assertIsNone(a.observe())
        for _ in range(36):
            self.assertIsNotNone(a.intervene(0, 0.0))
        self.assertIsNone(a.intervene(0, 0.0))

    def test_causalab_all_cells_scripted(self):
        from causalab.m3_agent_v2 import run_episode

        def scripted(prompt):
            match = re.search(r"Choose (\d+) interventions", prompt)
            n = int(match.group(1)) if match else 1
            return json.dumps({"menu_indices": list(range(n))})

        for acq, sub, menu in itertools.product(["llm", "engine"], ["llm", "engine"], ["screen", "neutral"]):
            row = run_episode(1000, acq, sub, menu, "offline-scripted", scripted)
            self.assertEqual(len(row["decision_log"]), 3)
            last = row["decision_log"][-1]
            self.assertEqual(len(last["menu"]), 6)
            self.assertEqual(last["engine"], 5)
            self.assertIsNotNone(last["llm"])
            self.assertIn(row["accuracy"], [0, 1])

    def test_block_estimator_and_rejections(self):
        from analyze import factorial, summarize
        rows = [dict(model="test", seed=s, experiment="test", acq=a, sub=b,
                     rate_itt=.5 + (.1 if a == "engine" else 0))
                for s in [0, 1] for a, b in itertools.product(["llm", "engine"], repeat=2)]
        result, _ = factorial(rows, "causalgame")
        np.testing.assert_allclose(result[("test", "acquisition")], [10, 10])
        self.assertEqual(summarize([0, 0])["decision"], "DELEGATE")
        with self.assertRaises(ValueError):
            factorial(rows[:-1], "causalgame")
        with self.assertRaises(ValueError):
            factorial(rows + [rows[0]], "causalgame")

    @unittest.skipUnless((ROOT / "external/causalgame/.mpi-prepared.json").is_file(),
                         "Download CausalGame with python prepare_data.py causalgame first")
    def test_backend_and_scripted_causalgame_episode(self):
        from prepare_data import use_causalgame
        use_causalgame()
        with patch.dict(os.environ, {"CAUSALGAME_SERVE_STATIC": "false", "ADMIN_TOKEN": "offline-test-only"}):
            from fastapi.testclient import TestClient
            from api.app import app
            from api.modules.agent.session import SessionManager
            from agent.client import CanyonClient
            from scripts.factorial_agent import run_cell
            import requests

            # TestClient runs the real app in-process: no network sockets or API keys.
            with tempfile.TemporaryDirectory(prefix="mpi-backend-test-") as temporary:
                previous = Path.cwd()
                os.chdir(temporary)
                try:
                    with patch.object(SessionManager, "_get_sessions_dir", return_value=Path(temporary) / "sessions"), TestClient(app) as server:
                        def get(url, **kwargs):
                            return server.get(url, **kwargs)

                        def post(url, **kwargs):
                            return server.post(url, **kwargs)

                        with patch.object(requests, "get", get), patch.object(requests, "post", post), patch(
                            "scripts.factorial_agent.make_llm", return_value=lambda _: '{"menu_index": 0}'
                        ):
                            for scene in ["antenna_trap", "antenna_trap_high_def", "antenna_trap_simpsons_paradox",
                                          "deployment_zone_trap_categorical", "deployment_zone_trap_env_shift", "weather_noise"]:
                                client = CanyonClient(base_url="http://testserver", experiment=scene)
                                self.assertTrue(client.get_action_space())
                                self.assertTrue(client.get_status())
                            client = CanyonClient(base_url="http://testserver", experiment="antenna_trap")
                            row = run_cell(client, "engine", "engine", "offline-scripted", 400, verbose=False)
                            self.assertTrue(row["decision_log"])
                            self.assertFalse(row["infra_errors"], row["infra_errors"])
                            self.assertIsNotNone(row["rate"])
                finally:
                    os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
