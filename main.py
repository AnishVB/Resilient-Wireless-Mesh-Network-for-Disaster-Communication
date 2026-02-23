#!/usr/bin/env python3
"""
Main entry point for Disaster-Resilient Wireless Mesh Network Simulator

This script orchestrates the complete simulation pipeline:
1. Network deployment and topology creation
2. LSTM model training for failure prediction
3. ARIMA-based link quality forecasting
4. Proactive rerouting with self-healing
5. Performance analysis and visualization

Usage:
    python main.py [--nodes=5] [--steps=100] [--no-plot]

Options:
    --nodes: Number of mesh nodes (default: 5)
    --steps: Number of simulation steps (default: 100)
    --no-plot: Skip plot generation (default: False)
"""

import argparse
import sys
from integrationbridge import NetworkBridge
from results import NetworkPerformanceAnalyzer
import json
from datetime import datetime


def print_header():
    """Print fancy header"""
    header = """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           DISASTER-RESILIENT WIRELESS MESH NETWORK SIMULATOR              ║
║                                                                            ║
║  A Comprehensive Study on AI-Driven Self-Healing Network Routing          ║
║  Using LSTM & ARIMA Time-Series Analysis                                  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(header)


def print_footer():
    """Print completion footer"""
    footer = """
╔════════════════════════════════════════════════════════════════════════════╗
║                     ✓ SIMULATION COMPLETED SUCCESSFULLY                   ║
║                                                                            ║
║  Review the following outputs:                                            ║
║  • results/ directory: Performance analysis plots                         ║
║  • simulation_report.json: Detailed metrics and statistics                ║
║  • Console output: Real-time network events and decisions                 ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(footer)


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
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        import numpy as np
        import random
        import tensorflow as tf
        np.random.seed(args.seed)
        random.seed(args.seed)
        tf.random.set_seed(args.seed)
        print(f"[CONFIG] Random seed set to {args.seed}")
    
    print_header()
    
    print(f"""
╭─ SIMULATION CONFIGURATION ─────────────────────────────────────────────╮
│                                                                        │
│  Number of Nodes:              {args.nodes}                                    │
│  Simulation Steps:             {args.steps}                                   │
│  Generate Plots:               {'Yes' if not args.no_plot else 'No':<43} │
│  Timestamp:                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}              │
│                                                                        │
╰────────────────────────────────────────────────────────────────────────╯
""")
    
    try:
        # Phase 1: Network Deployment & Simulation
        print("\n" + "="*78)
        print(" PHASE 1: NETWORK DEPLOYMENT & AI-DRIVEN SIMULATION")
        print("="*78)
        
        bridge = NetworkBridge(num_nodes=args.nodes, num_steps=args.steps)
        report = bridge.deploy_and_run()
        
        # Phase 2: Performance Analysis
        print("\n" + "="*78)
        print(" PHASE 2: PERFORMANCE ANALYSIS")
        print("="*78)
        
        metrics = report['simulation_metrics']
        rerouting = report['rerouting_analysis']
        
        analysis = f"""
╭─ KEY PERFORMANCE INDICATORS ──────────────────────────────────────────╮
│                                                                        │
│  Packet Delivery Ratio (PDR):                                         │
│    - Average:                  {metrics['avg_pdr']:.2f}%                              │
│    - Minimum:                  {metrics['min_pdr']:.2f}%                              │
│                                                                        │
│  Network Latency:                                                     │
│    - Average Delay:            {metrics['avg_delay']:.2f} ms                           │
│                                                                        │
│  Throughput:                                                          │
│    - Average:                  {metrics['avg_throughput']:.2f} Mbps                        │
│                                                                        │
│  Self-Healing Performance:                                            │
│    - Total Reroutes:           {metrics['total_reroutes']}                              │
│    - Successful:               {rerouting['successful_reroutes']}                              │
│    - Critical Prevented:       {rerouting['critical_reroutes']}                              │
│                                                                        │
│  Network Connectivity:                                                │
│    - Final Active Links:       {metrics['final_active_links']:.0f}                              │
│                                                                        │
╰────────────────────────────────────────────────────────────────────────╯
"""
        print(analysis)
        
        # Phase 3: Save Report
        print("\n" + "="*78)
        print(" PHASE 3: GENERATING REPORT")
        print("="*78)
        
        report['configuration'] = {
            'num_nodes': args.nodes,
            'num_steps': args.steps,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('simulation_report.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_report = {
                'configuration': report['configuration'],
                'simulation_metrics': report['simulation_metrics'],
                'rerouting_analysis': {
                    'total_reroutes': rerouting['total_reroutes'],
                    'successful_reroutes': rerouting['successful_reroutes'],
                    'critical_reroutes': rerouting['critical_reroutes']
                }
            }
            json.dump(json_report, f, indent=2)
        
        print("[SUCCESS] Report saved to: simulation_report.json")
        
        # Phase 4: Generate Visualizations
        if not args.no_plot:
            print("\n" + "="*78)
            print(" PHASE 4: GENERATING VISUALIZATIONS")
            print("="*78)
            
            analyzer = NetworkPerformanceAnalyzer(output_dir="results")
            analyzer.generate_comprehensive_report(report)
        else:
            print("\n[INFO] Plot generation skipped (--no-plot flag set)")
        
        print_footer()
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
