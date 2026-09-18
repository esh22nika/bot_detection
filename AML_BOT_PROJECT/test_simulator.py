import unittest
import json
from app import app, engine, data_loader

class TestBotDriftSimulator(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        engine.reset()

    def test_01_index_loads(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'BotDriftBench: Static vs Drift-Aware Detection', res.data)
        self.assertIn(b'performanceChart', res.data)
        self.assertIn(b'distributionChart', res.data)

    def test_02_initial_data(self):
        res = self.client.get('/api/initial-data')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['benchmark'], 'BotDriftBench v0.8')
        self.assertEqual(data['accounts'], 3000)
        self.assertEqual(len(data['trajectories']['windows']), 20)
        self.assertIn('content_similarity', data['features'])

    def test_03_before_drift_state(self):
        # First inject drift, then call before-drift
        self.client.post('/api/inject-drift')
        res = self.client.post('/api/before-drift')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['mode'], 'before_drift')
        self.assertEqual(data['current_window'], 0)
        self.assertFalse(data['drift_injected'])
        self.assertEqual(data['static_status'], 'stable')
        self.assertEqual(data['drift_aware_status'], 'stable')
        self.assertAlmostEqual(data['static_f1'], 0.9746, places=3)
        self.assertAlmostEqual(data['drift_aware_f1'], 0.9746, places=3)

    def test_04_inject_drift(self):
        res = self.client.post('/api/inject-drift')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['mode'], 'drift_injected')
        self.assertEqual(data['current_window'], 2)
        self.assertTrue(data['drift_injected'])
        self.assertFalse(data['drift_detected'])
        self.assertEqual(data['static_status'], 'degraded')
        self.assertEqual(data['drift_aware_status'], 'detecting')

    def test_05_run_adaptation(self):
        res = self.client.post('/api/run-adaptation')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['mode'], 'adapted')
        self.assertEqual(data['current_window'], 5)
        self.assertTrue(data['drift_injected'])
        self.assertTrue(data['drift_detected'])
        self.assertTrue(data['adaptation_triggered'])
        self.assertEqual(data['retraining_count'], 1)
        self.assertEqual(data['drift_aware_status'], 'adapted')
        # Drift-aware F1 > Static F1 at window 5
        self.assertGreater(data['drift_aware_f1'], data['static_f1'])

    def test_06_step_advances_one_window(self):
        engine.reset()
        self.assertEqual(engine.current_window, 0)
        
        # Step to window 1
        res = self.client.post('/api/step')
        data = res.get_json()
        self.assertEqual(data['current_window'], 1)
        
        # Step to window 2 (drift starts)
        res = self.client.post('/api/step')
        data = res.get_json()
        self.assertEqual(data['current_window'], 2)
        self.assertTrue(data['drift_injected'])

    def test_07_reset(self):
        # Advance multiple steps
        self.client.post('/api/run-adaptation')
        res = self.client.post('/api/reset')
        data = res.get_json()
        self.assertEqual(data['current_window'], 0)
        self.assertEqual(data['mode'], 'before_drift')
        self.assertFalse(data['drift_injected'])
        self.assertEqual(data['retraining_count'], 0)

    def test_08_events_and_timeline(self):
        self.client.post('/api/run-adaptation')
        res = self.client.get('/api/events')
        data = res.get_json()
        self.assertTrue(data['success'])
        event_windows = [e['window'] for e in data['events']]
        self.assertIn(0, event_windows)  # Init
        self.assertIn(2, event_windows)  # Drift onset
        self.assertIn(5, event_windows)  # Detection & Adaptation

    def test_09_feature_distribution_endpoint(self):
        res = self.client.get('/api/features/content_similarity')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('pre_drift', data['distribution'])
        self.assertIn('post_drift', data['distribution'])
        self.assertEqual(len(data['distribution']['pre_drift']), 10)

    def test_10_invalid_requests_return_json_errors(self):
        # Non-existent endpoint
        res = self.client.get('/api/non-existent-endpoint')
        self.assertEqual(res.status_code, 404)
        data = res.get_json()
        self.assertFalse(data['success'])
        self.assertIn('error', data)

        # Invalid feature name
        res = self.client.get('/api/features/invalid_feature_xyz')
        self.assertEqual(res.status_code, 404)
        data = res.get_json()
        self.assertFalse(data['success'])

        # Method not allowed (GET on POST-only endpoint)
        res = self.client.get('/api/inject-drift')
        self.assertEqual(res.status_code, 405)
        data = res.get_json()
        self.assertFalse(data['success'])

if __name__ == '__main__':
    unittest.main()
