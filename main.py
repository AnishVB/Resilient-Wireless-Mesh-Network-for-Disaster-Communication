#!/usr/bin/env python3
"""
Disaster-Resilient Wireless Mesh Network Simulator

Usage:
    python main.py [--nodes=5] [--steps=100] [--no-plot] [--visualize] [--seed=42]
"""

import argparse
import sys
from integrationbridge import NetworkBridge
from results import NetworkPerformanceAnalyzer
from visualization import MeshNetworkVisualizer
import json
from datetime import datetime
import numpy as np


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    """Main execution pipeline"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Disaster-Resilient Wireless Mesh Network Simulator'
    )
    parser.add_argument('--nodes', type=int, default=5, 
                       help='Number of mesh nodes (default: 5)')
    parser.add_argument('--steps', type=int, default=100,
                       help='Number of simulation steps (default: 100)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true',
                       help='Show live network visualization')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        import numpy as np
        import random
        import tensorflow as tf
        np.random.seed(args.seed)
        random.seed(args.seed)
        tf.random.set_seed(args.seed)
    
    print("Disaster-Resilient Mesh Network Simulator")
    print(f"Nodes: {args.nodes}, Steps: {args.steps}")
    print()
    
    try:
        # Phase 1: Network Deployment & Simulation
        print("Running simulation...")
        
        bridge = NetworkBridge(num_nodes=args.nodes, num_steps=args.steps)
        report = bridge.deploy_and_run()
        
        # Phase 2: Performance Analysis
        metrics = report['simulation_metrics']
        rerouting = report['rerouting_analysis']
        
        print("\nResults:")
        print(f"Average PDR: {metrics['avg_pdr']:.2f}%")
        print(f"Minimum PDR: {metrics['min_pdr']:.2f}%")
        print(f"Average Delay: {metrics['avg_delay']:.2f} ms")
        print(f"Average Throughput: {metrics['avg_throughput']:.2f} Mbps")
        print(f"Total Reroutes: {metrics['total_reroutes']}")
        print(f"Successful Reroutes: {rerouting['successful_reroutes']}")
        print(f"Final Active Links: {metrics['final_active_links']:.0f}")
        
        # Phase 3: Save Report
        print("\nSaving report...")
        with open('simulation_report.json', 'w') as f:
            json_report = {
                'timestamp': datetime.now().isoformat(),
                'configuration': {
                    'nodes': args.nodes,
                    'steps': args.steps,
                    'seed': args.seed
                },
                'simulation_metrics': report['simulation_metrics'],
                'rerouting_analysis': {
                    'total_reroutes': rerouting['total_reroutes'],
                    'successful_reroutes': rerouting['successful_reroutes'],
                    'critical_reroutes': rerouting['critical_reroutes']
                }
            }
            # Convert numpy types to Python native types
            json_report = convert_to_serializable(json_report)
            json.dump(json_report, f, indent=2)
        
        print("Report saved to: simulation_report.json")
        
        # Phase 4: Live Visualization (if requested)
        if args.visualize:
            print("\nShowing network visualization...")
            visualizer = MeshNetworkVisualizer(bridge.simulator, 
                                               "Disaster-Resilient Mesh Network - Live Simulation")
            visualizer.show_current_state()
            visualizer.show_interactive_dashboard(report['history'])
            print("Close windows to continue")
        
        # Phase 5: Generate Static Plots
        if not args.no_plot:
            print("\nGenerating plots...")
            analyzer = NetworkPerformanceAnalyzer(output_dir="results")
            analyzer.generate_comprehensive_report(report)
            print("Plots saved to results/")
        
        print("\nSimulation complete!")
        
    except Exception as e:
        print(f"\nError during simulation: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())