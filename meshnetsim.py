import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time
import random
from scipy.spatial.distance import euclidean
from failprediction import NodeFailurePredictor

@dataclass
class Node:
    """Represents a mesh network node (drone/rescue team)"""
    node_id: int
    x: float
    y: float
    z: float
    is_alive: bool = True
    signal_strength: float = -60.0  # RSSI in dBm
    
    def position(self):
        return (self.x, self.y, self.z)
    
    def distance_to(self, other: 'Node') -> float:
        """Calculate 3D distance to another node"""
        return euclidean(self.position(), other.position())


@dataclass
class Link:
    """Represents a wireless link between two nodes"""
    source_id: int
    dest_id: int
    rssi: float  # Signal strength
    link_quality: float  # 0-100 (%)
    hop_count: int
    is_active: bool = True
    reroute_count: int = 0
    
    def is_healthy(self, threshold: float = 70.0) -> bool:
        """Check if link quality is above threshold"""
        return self.link_quality >= threshold


class MeshNetworkSimulator:
    """
    Simulates a disaster-resilient wireless mesh network
    with AODV/OLSR-like routing and self-healing capabilities
    """
    
    def __init__(self, num_nodes: int = 5, simulation_area: Tuple[float, float, float] = (100, 100, 50)):
        self.num_nodes = num_nodes
        self.simulation_area = simulation_area
        self.nodes: Dict[int, Node] = {}
        self.links: Dict[Tuple[int, int], Link] = {}
        self.time_step = 0
        self.failure_predictor = NodeFailurePredictor()
        self.history = {
            'time': [],
            'packet_delivery_ratio': [],
            'delay': [],
            'throughput': [],
            'active_links': [],
            'reroutes_triggered': [],
            'predicted_failures': [],
            'link_quality': {}
        }
        self.reroute_events = []
        
    def initialize_nodes(self):
        """Create nodes at random positions in the disaster area"""
        print("[INIT] Initializing mesh nodes...")
        positions = self._generate_node_positions()
        
        for i in range(self.num_nodes):
            node = Node(
                node_id=i,
                x=positions[i][0],
                y=positions[i][1],
                z=positions[i][2]
            )
            self.nodes[i] = node
            print(f"  Node {i}: Position ({node.x:.1f}, {node.y:.1f}, {node.z:.1f})")
    
    def _generate_node_positions(self) -> List[Tuple[float, float, float]]:
        """Generate random node positions (can be customized for specific topologies)"""
        positions = []
        for _ in range(self.num_nodes):
            x = random.uniform(0, self.simulation_area[0])
            y = random.uniform(0, self.simulation_area[1])
            z = random.uniform(0, self.simulation_area[2])
            positions.append((x, y, z))
        return positions
    
    def _calculate_rssi(self, distance: float) -> float:
        """
        Calculate RSSI based on distance (path loss model)
        RSSI = -30 - 20*log10(distance) + random noise
        """
        if distance == 0:
            return -30.0
        
        # Free space path loss model
        base_rssi = -30 - 20 * np.log10(max(distance, 1))
        
        # Add fading and noise
        noise = np.random.normal(0, 3)  # 3dB standard deviation
        rssi = base_rssi + noise
        
        # Clamp to realistic range
        return np.clip(rssi, -100, -30)
    
    def _calculate_link_quality(self, rssi: float) -> float:
        """
        Convert RSSI to link quality percentage
        -100 dBm = 0%, -30 dBm = 100%
        """
        quality = ((rssi + 100) / 70) * 100  # Normalize to 0-100%
        return np.clip(quality, 0, 100)
    
    def build_topology(self):
        """Build mesh topology based on node distances (AODV)"""
        print("[TOPO] Building mesh topology...")
        self.links.clear()
        
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                node_i = self.nodes[i]
                node_j = self.nodes[j]
                
                # Calculate distance and RSSI
                distance = node_i.distance_to(node_j)
                rssi = self._calculate_rssi(distance)
                quality = self._calculate_link_quality(rssi)
                
                # Links within communication range (RSSI > -90 dBm)
                if rssi > -90:
                    link_ij = Link(
                        source_id=i,
                        dest_id=j,
                        rssi=rssi,
                        link_quality=quality,
                        hop_count=1
                    )
                    link_ji = Link(
                        source_id=j,
                        dest_id=i,
                        rssi=rssi,
                        link_quality=quality,
                        hop_count=1
                    )
                    
                    self.links[(i, j)] = link_ij
                    self.links[(j, i)] = link_ji
    
    def simulate_step(self, failure_probability: float = 0.05, degrade_probability: float = 0.1):
        """
        Simulate one timestep of network operation
        - Update RSSI values
        - Detect link failures
        - Apply LSTM predictions
        - Trigger reroutes if needed
        """
        self.time_step += 1
        self.history['time'].append(self.time_step)
        
        # Update link qualities and detect degradation
        links_to_remove = []
        predicted_failures = 0
        
        for link_key, link in self.links.items():
            if not link.is_active:
                continue
            
            # Simulate RSSI fluctuations
            rssi_change = np.random.normal(0, 2)
            link.rssi += rssi_change
            link.rssi = np.clip(link.rssi, -100, -30)
            link.link_quality = self._calculate_link_quality(link.rssi)
            
            # Random link failures (disaster events)
            if random.random() < failure_probability:
                links_to_remove.append(link_key)
                continue
            
            # Link degradation
            if random.random() < degrade_probability and link.link_quality > 0:
                link.link_quality -= random.uniform(5, 15)
                link.rssi -= 5
            
            # LSTM-based failure prediction
            if self.failure_predictor.trained:
                rssi_normalized = (link.rssi + 100) / 100
                # Build a sequence (simplified with repeated values for demo)
                rssi_sequence = [rssi_normalized] * 10
                probability = self.failure_predictor.predict_failure_probability(rssi_sequence)
                
                if probability > 0.75:
                    predicted_failures += 1
                    link.reroute_count += 1
                    self.reroute_events.append({
                        'time': self.time_step,
                        'link': link_key,
                        'predicted_prob': probability
                    })
        
        # Remove failed links
        for link_key in links_to_remove:
            del self.links[link_key]
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        self.history['packet_delivery_ratio'].append(metrics['pdr'])
        self.history['delay'].append(metrics['delay'])
        self.history['throughput'].append(metrics['throughput'])
        self.history['active_links'].append(metrics['active_links'])
        self.history['reroutes_triggered'].append(predicted_failures)
        self.history['predicted_failures'].append(predicted_failures)
        
        return metrics
    
    def _calculate_metrics(self) -> Dict:
        """Calculate network performance metrics"""
        active_links = len([l for l in self.links.values() if l.is_active])
        max_links = self.num_nodes * (self.num_nodes - 1) / 2
        
        # Packet Delivery Ratio (affected by link quality)
        if active_links > 0:
            avg_quality = np.mean([l.link_quality for l in self.links.values() if l.is_active])
            pdr = (active_links / max_links) * (avg_quality / 100) * 100
        else:
            pdr = 0
        
        # Delay (hops-based): typical WiFi delay ~5ms per hop
        avg_hops = np.mean([l.hop_count for l in self.links.values() if l.is_active]) if active_links > 0 else 0
        delay = avg_hops * 5 + np.random.uniform(1, 3)  # Add queueing delay
        
        # Throughput (affected by link quality): max 54 Mbps for 802.11g mesh
        throughput = (avg_quality / 100 * 54 if active_links > 0 else 0) + np.random.uniform(-5, 5)
        throughput = max(0, throughput)
        
        return {
            'pdr': np.clip(pdr, 0, 100),
            'delay': max(0, delay),
            'throughput': max(0, throughput),
            'active_links': active_links,
            'total_links': max_links
        }
    
    def get_connectivity_map(self) -> Dict:
        """
        Generate self-healing connectivity map showing:
        - Active/inactive nodes
        - Link states
        - Reroute recommendations
        """
        connectivity_map = {
            'nodes': {},
            'links': {},
            'reroute_paths': {},
            'timestamp': self.time_step
        }
        
        # Node status
        for node_id, node in self.nodes.items():
            connectivity_map['nodes'][node_id] = {
                'position': node.position(),
                'status': 'alive' if node.is_alive else 'dead',
                'signal_strength': node.signal_strength
            }
        
        # Link status
        for link_key, link in self.links.items():
            connectivity_map['links'][link_key] = {
                'rssi': link.rssi,
                'quality': link.link_quality,
                'active': link.is_active,
                'reroutes': link.reroute_count
            }
        
        # Reroute recommendations (alternative paths)
        for src_id in self.nodes:
            for dst_id in self.nodes:
                if src_id != dst_id:
                    path = self._find_backup_path(src_id, dst_id)
                    if path:
                        connectivity_map['reroute_paths'][(src_id, dst_id)] = path
        
        return connectivity_map
    
    def _find_backup_path(self, source: int, destination: int) -> List[int]:
        """Find alternative path for rerouting (simplified BFS)"""
        visited = set()
        queue = [(source, [source])]
        
        while queue and len(queue[0][1]) < self.num_nodes:
            current, path = queue.pop(0)
            
            if current == destination:
                return path
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for link_key, link in self.links.items():
                if link_key[0] == current and link_key[1] not in visited:
                    queue.append((link_key[1], path + [link_key[1]]))
        
        return None
    
    def run_simulation(self, steps: int = 100, enable_ml_prediction: bool = True):
        """Run complete simulation"""
        print("\n[SIM] Initializing simulation...")
        self.initialize_nodes()
        self.build_topology()
        
        # Train LSTM model if ML enabled
        if enable_ml_prediction:
            print("[ML] Training failure prediction model...")
            self.failure_predictor.train(epochs=50)
            self.failure_predictor.save_model()
        
        print(f"\n[SIM] Running simulation for {steps} timesteps...")
        print(f"[SIM] Network has {self.num_nodes} nodes\n")
        
        for step in range(steps):
            metrics = self.simulate_step()
            
            if (step + 1) % 10 == 0:
                print(f"Step {step+1}: PDR={metrics['pdr']:.1f}% | "
                      f"Links={metrics['active_links']:.0f} | "
                      f"Delay={metrics['delay']:.1f}ms | "
                      f"Throughput={metrics['throughput']:.1f}Mbps")
        
        print("\n[SUCCESS] Simulation complete!")
        return self.history


# Example usage
if __name__ == "__main__":
    # Create simulator
    sim = MeshNetworkSimulator(num_nodes=5, simulation_area=(100, 100, 50))
    
    # Run simulation
    history = sim.run_simulation(steps=100, enable_ml_prediction=True)
    
    # Get connectivity map
    connectivity_map = sim.get_connectivity_map()
    print("\n[MAP] Connectivity Map generated with reroute paths")