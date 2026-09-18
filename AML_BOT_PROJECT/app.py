import os
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from simulator.data_loader import DataLoader
from simulator.engine import SimulatorEngine

app = Flask(__name__, template_folder='templates', static_folder='static')

# Initialize shared data loader and stateful simulator engine
data_loader = DataLoader()
engine = SimulatorEngine(data_loader)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/initial-data', methods=['GET'])
def get_initial_data():
    try:
        trajectories = data_loader.get_all_trajectories()
        features = data_loader.get_feature_list()
        default_feature = features[0] if features else 'content_similarity'
        feature_dist = data_loader.get_feature_distribution(default_feature)
        
        return jsonify({
            'success': True,
            'benchmark': 'BotDriftBench v0.8',
            'accounts': 3000,
            'test_accounts': 600,
            'windows': 20,
            'trajectories': trajectories,
            'features': features,
            'default_feature': default_feature,
            'feature_distribution': feature_dist,
            'initial_state': engine.get_state()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/state', methods=['GET'])
def get_state():
    try:
        return jsonify(engine.get_state())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    try:
        return jsonify(engine.reset())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/before-drift', methods=['POST'])
def before_drift():
    try:
        return jsonify(engine.before_drift())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/inject-drift', methods=['POST'])
def inject_drift():
    try:
        return jsonify(engine.inject_drift())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/run-adaptation', methods=['POST'])
def run_adaptation():
    try:
        return jsonify(engine.run_adaptation())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/step', methods=['POST'])
def step():
    try:
        return jsonify(engine.step())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/comparison', methods=['GET'])
def get_comparison():
    try:
        return jsonify(engine.get_comparison())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/events', methods=['GET'])
def get_events():
    try:
        return jsonify(engine.get_events())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/features/<feature_name>', methods=['GET'])
def get_feature_distribution(feature_name):
    try:
        dist = data_loader.get_feature_distribution(feature_name)
        if not dist:
            return jsonify({'success': False, 'error': f"Feature '{feature_name}' not found."}), 404
        return jsonify({'success': True, 'feature': feature_name, 'distribution': dist})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(400)
def bad_request(e):
    return jsonify({'success': False, 'error': 'Bad Request', 'message': str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found', 'message': str(e)}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'success': False, 'error': 'Method not allowed for this endpoint', 'message': str(e)}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
