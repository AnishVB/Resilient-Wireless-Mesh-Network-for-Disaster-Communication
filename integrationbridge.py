import numpy as np
import time
from collections import deque
from typing import List, Dict
from failprediction import NodeFailurePredictor
from meshnetsim import MeshNetworkSimulator, Link
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')


class LinkQualityForecaster:
    """
    ARIMA-based forecasting for link quality prediction
    Time-series analysis of link stability
    """
    
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.models = {}
        self.history_length = 30
        
    def fit_link_forecast(self, link_key: tuple, quality_history: List[float]):
        """Fit ARIMA model for a specific link"""
        if len(quality_history) < 10:
            return None
        
        try:
            data = quality_history[-self.history_length:]
            model = ARIMA(data, order=self.order)
            fitted_model = model.fit()
            self.models[link_key] = fitted_model
            return fitted_model
        except Exception as e:
            return None
    
    def forecast_link_quality(self, link_key: tuple, steps: int = 5) -> List[float]:
        """Forecast future link quality"""
        if link_key not in self.models:
            return None
        
        try:
            forecast = self.models[link_key].get_forecast(steps=steps)
            return forecast.predicted_mean.tolist()
        except Exception as e:
            return None


class ProactiveReroutingEngine:
    """
    Self-healing routing engine that:
    1. Monitors link quality
    2. Predicts failures using LSTM
    3. Forecasts trends using ARIMA
    4. Triggers proactive reroutes BEFORE failures
    """
    
    def __init__(self, simulator: MeshNetworkSimulator, threshold: float = 0.75):
        self.simulator = simulator
        self.predictor = simulator.failure_predictor
        self.forecaster = LinkQualityForecaster()
        self.threshold = threshold
        self.link_history = {}
        self.reroute_log = []
        self.active_reroutes = {}
        self.max_history = 30
        
    def monitor_link_health(self):
        """Continuously monitor and analyze link health"""
        alerts = []
        
        for link_key, link in self.simulator.links.items():
            if not link.is_active:
                continue
            
            # Initialize history for this link
            if link_key not in self.link_history:
                self.link_history[link_key] = deque(maxlen=self.max_history)
            
            self.link_history[link_key].append(link.link_quality)
            
            # Only analyze after sufficient history
            if len(self.link_history[link_key]) < 5:
                continue
            
            # 1. Check current link quality
            if link.link_quality < 50:
                alerts.append({
                    'type': 'CRITICAL_QUALITY',
                    'link': link_key,
                    'quality': link.link_quality,
                    'action': 'IMMEDIATE_REROUTE'
                })
                continue
            
            # 2. LSTM-based failure prediction
            quality_history = list(self.link_history[link_key])
            if len(quality_history) >= 10:
                rssi_history = [(q / 100) for q in quality_history[-10:]]  # Simplified
                
                try:
                    failure_prob = self.predictor.predict_failure_probability(np.array(rssi_history))
                    
                    if failure_prob > self.threshold:
                        alerts.append({
                            'type': 'PREDICTED_FAILURE',
                            'link': link_key,
                            'probability': failure_prob,
                            'action': 'PROACTIVE_REROUTE'
                        })
                except Exception as e:
                    pass
            
            # 3. ARIMA-based trend analysis
            if len(quality_history) >= 10:
                self.forecaster.fit_link_forecast(link_key, quality_history)
                forecast = self.forecaster.forecast_link_quality(link_key, steps=5)
                
                if forecast:
                    # If forecast predicts collapse within next 5 steps
                    if min(forecast) < 30:
                        trend_slope = forecast[-1] - forecast[0]
                        if trend_slope < -10:  # Rapid degradation
                            alerts.append({
                                'type': 'TREND_DEGRADATION',
                                'link': link_key,
                                'forecast': forecast,
                                'action': 'PREVENTIVE_REROUTE',
                                'severity': 'HIGH' if trend_slope < -20 else 'MEDIUM'
                            })
        
        return alerts
    
    def execute_proactive_reroute(self, alert: Dict) -> bool:
        """
        Execute proactive reroute BEFORE link failure
        Returns: True if successful, False otherwise
        """
        link_key = alert['link']
        src, dst = link_key
        
        print(f"\n[REROUTE] Alert Type: {alert['type']}")
        print(f"          Link: {src} -> {dst}")
        print(f"          Action: {alert['action']}")
        
        # Find alternative path
        alt_path = self.simulator._find_backup_path(src, dst)
        
        if alt_path and len(alt_path) > 2:
            print(f"          Alternative Path: {' -> '.join(map(str, alt_path))}")
            
            event = {
                'timestamp': self.simulator.time_step,
                'original_link': link_key,
                'reroute_path': alt_path,
                'reason': alert['type'],
                'severity': alert.get('severity', 'NORMAL')
            }
            
            self.reroute_log.append(event)
            self.active_reroutes[link_key] = event
            
            print(f"          [SUCCESS] Reroute activated!\n")
            return True
        else:
            print(f"          [FAILED] No alternative path available\n")
            return False
    
    def run_proactive_loop(self, simulator_steps: int = 100):
        """Main control loop for proactive rerouting"""
        print("\n[BRIDGE] Proactive Rerouting Engine Started")
        print("[BRIDGE] Monitoring for link failures...\n")
        
        for step in range(simulator_steps):
            # Run simulation step
            self.simulator.simulate_step()
            
            # Monitor link health
            alerts = self.monitor_link_health()
            
            # Process alerts
            critical_count = len([a for a in alerts if a['action'] == 'IMMEDIATE_REROUTE'])
            proactive_count = len([a for a in alerts if a['action'] in ['PROACTIVE_REROUTE', 'PREVENTIVE_REROUTE']])
            
            if critical_count > 0 or proactive_count > 0:
                print(f"[BRIDGE] Step {step}: {critical_count} critical, {proactive_count} proactive alerts")
                
                # Execute reroutes in order of severity
                for alert in sorted(alerts, key=lambda x: x['action'] == 'IMMEDIATE_REROUTE', reverse=True):
                    self.execute_proactive_reroute(alert)
    
    def get_rerouting_report(self) -> Dict:
        """Generate detailed rerouting report"""
        return {
            'total_reroutes': len(self.reroute_log),
            'successful_reroutes': len([r for r in self.reroute_log if r['severity'] != 'FAILED']),
            'critical_reroutes': len([r for r in self.reroute_log if r['severity'] == 'HIGH']),
            'reroute_events': self.reroute_log,
            'active_reroutes': self.active_reroutes
        }


class NetworkBridge:
    """
    Integration bridge combining:
    - LSTM failure prediction (failprediction.py)
    - Mesh network simulator (meshnetsim.py)
    - ARIMA link forecasting
    - Proactive rerouting engine
    """
    
    def __init__(self, num_nodes: int = 5, num_steps: int = 100):
        self.num_nodes = num_nodes
        self.num_steps = num_steps
        self.simulator = MeshNetworkSimulator(num_nodes=num_nodes)
        self.rerouting_engine = ProactiveReroutingEngine(self.simulator)
        
    def deploy_and_run(self):
        """Deploy mesh network and run disaster scenario"""
        print("="*70)
        print(" DISASTER-RESILIENT WIRELESS MESH NETWORK SIMULATOR")
        print("="*70)
        
        # Initialize simulator
        print("\n[DEPLOY] Step 1: Initialize Mesh Network")
        self.simulator.initialize_nodes()
        self.simulator.build_topology()
        
        # Train ML model
        print("\n[DEPLOY] Step 2: Train LSTM Failure Predictor")
        self.simulator.failure_predictor.train(epochs=50)
        self.simulator.failure_predictor.save_model()
        
        # Run proactive rerouting loop
        print("\n[DEPLOY] Step 3: Execute Network with AI-Driven Self-Healing")
        self.rerouting_engine.run_proactive_loop(self.num_steps)
        
        # Generate reports
        print("\n[DEPLOY] Step 4: Analyze Results")
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'simulation_metrics': self._extract_metrics(),
            'rerouting_analysis': self.rerouting_engine.get_rerouting_report(),
            'connectivity_map': self.simulator.get_connectivity_map(),
            'history': self.simulator.history,
            'link_forecasts': dict(self.rerouting_engine.forecaster.models)
        }
        return report
    
    def _extract_metrics(self) -> Dict:
        """Extract key performance metrics from simulation"""
        history = self.simulator.history
        
        return {
            'avg_pdr': np.mean(history['packet_delivery_ratio']),
            'min_pdr': np.min(history['packet_delivery_ratio']),
            'avg_delay': np.mean(history['delay']),
            'avg_throughput': np.mean(history['throughput']),
            'total_reroutes': np.sum(history['reroutes_triggered']),
            'final_active_links': history['active_links'][-1] if history['active_links'] else 0
        }


# Example usage
if __name__ == "__main__":
    # Create and run bridge
    bridge = NetworkBridge(num_nodes=5, num_steps=100)
    report = bridge.deploy_and_run()
    
    # Print summary
    print("\n" + "="*70)
    print(" SIMULATION SUMMARY")
    print("="*70)
    print(f"Average PDR: {report['simulation_metrics']['avg_pdr']:.2f}%")
    print(f"Average Delay: {report['simulation_metrics']['avg_delay']:.2f} ms")
    print(f"Average Throughput: {report['simulation_metrics']['avg_throughput']:.2f} Mbps")
    print(f"Total Reroutes Triggered: {report['simulation_metrics']['total_reroutes']}")
    print(f"Rerouting Report: {report['rerouting_analysis']['total_reroutes']} reroutes executed")
    print("="*70)