import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import numpy as np
from typing import Dict, List
import matplotlib.patches as mpatches

class MeshNetworkVisualizer:
    """
    Real-time visualization of mesh network simulation
    Shows nodes, links, and link quality in an interactive window
    """
    
    def __init__(self, simulator, title="Disaster-Resilient Mesh Network Simulator"):
        self.simulator = simulator
        self.title = title
        self.fig = None
        self.ax = None
        self.frames_data = []
        
    def plot_static_frame(self, step: int = None):
        """
        Create a single visualization frame showing current network state
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Set up the plot
        ax.set_xlim(-10, self.simulator.simulation_area[0] + 10)
        ax.set_ylim(-10, self.simulator.simulation_area[1] + 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw links with color based on quality
        for link_key, link in self.simulator.links.items():
            if not link.is_active:
                continue
            
            src_id, dst_id = link_key
            src_node = self.simulator.nodes[src_id]
            dst_node = self.simulator.nodes[dst_id]
            
            # Color based on link quality
            if link.link_quality > 80:
                color = 'green'
                alpha = 0.8
                linewidth = 2.5
            elif link.link_quality > 60:
                color = 'orange'
                alpha = 0.6
                linewidth = 2
            elif link.link_quality > 40:
                color = 'red'
                alpha = 0.5
                linewidth = 1.5
            else:
                color = 'darkred'
                alpha = 0.3
                linewidth = 1
            
            # Draw link line
            ax.plot([src_node.x, dst_node.x], 
                   [src_node.y, dst_node.y],
                   color=color, linewidth=linewidth, alpha=alpha, zorder=1)
            
            # Add RSSI label at midpoint
            mid_x = (src_node.x + dst_node.x) / 2
            mid_y = (src_node.y + dst_node.y) / 2
            ax.text(mid_x, mid_y, f'{link.link_quality:.0f}%', 
                   fontsize=8, ha='center', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.7))
        
        # Draw nodes
        for node_id, node in self.simulator.nodes.items():
            if node.is_alive:
                color = 'lightblue'
                marker = 'o'
                size = 500
                edge_color = 'darkblue'
                edge_width = 2
            else:
                color = 'lightgray'
                marker = 'x'
                size = 400
                edge_color = 'black'
                edge_width = 1.5
            
            scatter = ax.scatter(node.x, node.y, s=size, c=color, 
                               marker=marker, edgecolors=edge_color, 
                               linewidth=edge_width, zorder=2)
            
            # Add node label
            ax.text(node.x, node.y - 3, f'Node {node_id}', 
                   fontsize=10, ha='center', fontweight='bold')
            
            # Add RSSI value
            ax.text(node.x, node.y + 3, f'{node.signal_strength:.0f}dBm', 
                   fontsize=8, ha='center', style='italic')
        
        # Add title and info
        title_text = f"{self.title}\nTime Step: {self.simulator.time_step}"
        if step is not None:
            title_text = f"{self.title}\nStep: {step}"
        
        ax.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (meters)', fontsize=11)
        ax.set_ylabel('Y Position (meters)', fontsize=11)
        
        # Add legend
        green_patch = mpatches.Patch(color='green', label='Excellent (>80%)')
        orange_patch = mpatches.Patch(color='orange', label='Good (60-80%)')
        red_patch = mpatches.Patch(color='red', label='Poor (40-60%)')
        darkred_patch = mpatches.Patch(color='darkred', label='Critical (<40%)')
        blue_patch = mpatches.Patch(color='lightblue', label='Active Node')
        gray_patch = mpatches.Patch(color='lightgray', label='Dead Node')
        
        ax.legend(handles=[green_patch, orange_patch, red_patch, darkred_patch, 
                          blue_patch, gray_patch],
                 loc='upper right', fontsize=10, framealpha=0.95)
        
        # Add metrics box
        if self.simulator.history['time']:
            pdr = self.simulator.history['packet_delivery_ratio'][-1]
            delay = self.simulator.history['delay'][-1]
            throughput = self.simulator.history['throughput'][-1]
            active_links = self.simulator.history['active_links'][-1]
            
            metrics_text = f"""
NETWORK METRICS (Current Step)
────────────────────────────
PDR: {pdr:.1f}%
Delay: {delay:.2f} ms
Throughput: {throughput:.2f} Mbps
Active Links: {active_links:.0f}
            """
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   family='monospace')
        
        plt.tight_layout()
        return fig, ax
    
    def show_current_state(self):
        """Display current network state in a window"""
        fig, ax = self.plot_static_frame()
        plt.show()
        return fig, ax
    
    def create_animation(self, frames_list: List[Dict]):
        """
        Create animation from list of simulation frames
        frames_list: List of dictionaries with network state at each step
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        def animate(frame_idx):
            ax.clear()
            
            if frame_idx >= len(frames_list):
                return
            
            frame_data = frames_list[frame_idx]
            
            # Set up plot
            ax.set_xlim(-10, self.simulator.simulation_area[0] + 10)
            ax.set_ylim(-10, self.simulator.simulation_area[1] + 10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Draw links
            for link_key, link_info in frame_data.get('links', {}).items():
                src_id, dst_id = link_key
                src_node = frame_data['nodes'][src_id]
                dst_node = frame_data['nodes'][dst_id]
                
                quality = link_info['quality']
                
                if quality > 80:
                    color = 'green'
                    linewidth = 2.5
                elif quality > 60:
                    color = 'orange'
                    linewidth = 2
                elif quality > 40:
                    color = 'red'
                    linewidth = 1.5
                else:
                    color = 'darkred'
                    linewidth = 1
                
                ax.plot([src_node['x'], dst_node['x']], 
                       [src_node['y'], dst_node['y']],
                       color=color, linewidth=linewidth, alpha=0.6, zorder=1)
            
            # Draw nodes
            for node_id, node_info in frame_data['nodes'].items():
                ax.scatter(node_info['x'], node_info['y'], s=500, 
                          c='lightblue', edgecolors='darkblue', 
                          linewidth=2, zorder=2)
                ax.text(node_info['x'], node_info['y'] - 3, f"Node {node_id}", 
                       fontsize=10, ha='center', fontweight='bold')
            
            ax.set_title(f"{self.title} - Simulation Step {frame_idx}", 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('X Position', fontsize=11)
            ax.set_ylabel('Y Position', fontsize=11)
        
        anim = animation.FuncAnimation(fig, animate, frames=len(frames_list),
                                      interval=500, repeat=True)
        return fig, anim
    
    def show_metrics_over_time(self, history: Dict):
        """Display performance metrics over time in subplots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Network Performance Over Time', fontsize=16, fontweight='bold')
        
        time = history['time']
        
        # PDR
        axes[0, 0].plot(time, history['packet_delivery_ratio'], 'g-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_ylabel('PDR (%)', fontweight='bold')
        axes[0, 0].set_title('Packet Delivery Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 105])
        
        # Delay
        axes[0, 1].plot(time, history['delay'], 'b-', linewidth=2, marker='s', markersize=4)
        axes[0, 1].set_ylabel('Delay (ms)', fontweight='bold')
        axes[0, 1].set_title('Network Latency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Throughput
        axes[1, 0].plot(time, history['throughput'], 'r-', linewidth=2, marker='^', markersize=4)
        axes[1, 0].set_xlabel('Time (steps)', fontweight='bold')
        axes[1, 0].set_ylabel('Throughput (Mbps)', fontweight='bold')
        axes[1, 0].set_title('Network Throughput')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Active Links
        axes[1, 1].plot(time, history['active_links'], 'purple', linewidth=2, marker='D', markersize=4)
        axes[1, 1].set_xlabel('Time (steps)', fontweight='bold')
        axes[1, 1].set_ylabel('Count', fontweight='bold')
        axes[1, 1].set_title('Active Links')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def show_interactive_dashboard(self, history: Dict):
        """
        Show an interactive dashboard with network state and metrics
        """
        fig = plt.figure(figsize=(18, 10))
        
        # Create grid for subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Main network visualization
        ax_network = fig.add_subplot(gs[:, 0:2])
        
        # Metrics subplots
        ax_pdr = fig.add_subplot(gs[0, 2])
        ax_delay = fig.add_subplot(gs[1, 2])
        
        # Draw network
        ax_network.set_xlim(-10, self.simulator.simulation_area[0] + 10)
        ax_network.set_ylim(-10, self.simulator.simulation_area[1] + 10)
        ax_network.set_aspect('equal')
        ax_network.grid(True, alpha=0.3)
        
        # Draw links
        for link_key, link in self.simulator.links.items():
            if not link.is_active:
                continue
            
            src_id, dst_id = link_key
            src_node = self.simulator.nodes[src_id]
            dst_node = self.simulator.nodes[dst_id]
            
            quality = link.link_quality
            if quality > 80:
                color = 'green'
            elif quality > 60:
                color = 'orange'
            elif quality > 40:
                color = 'red'
            else:
                color = 'darkred'
            
            ax_network.plot([src_node.x, dst_node.x],
                          [src_node.y, dst_node.y],
                          color=color, linewidth=2, alpha=0.7, zorder=1)
        
        # Draw nodes
        for node_id, node in self.simulator.nodes.items():
            ax_network.scatter(node.x, node.y, s=400, c='lightblue',
                             edgecolors='darkblue', linewidth=2, zorder=2)
            ax_network.text(node.x, node.y, str(node_id), ha='center', va='center',
                          fontweight='bold', fontsize=12)
        
        ax_network.set_title('Mesh Network Topology', fontweight='bold', fontsize=12)
        ax_network.set_xlabel('X Position (m)')
        ax_network.set_ylabel('Y Position (m)')
        
        # PDR chart
        time = history['time']
        ax_pdr.plot(time, history['packet_delivery_ratio'], 'g-', linewidth=2)
        ax_pdr.fill_between(time, history['packet_delivery_ratio'], alpha=0.3, color='green')
        ax_pdr.set_ylabel('PDR (%)', fontweight='bold')
        ax_pdr.set_title('Packet Delivery Ratio', fontsize=10, fontweight='bold')
        ax_pdr.grid(True, alpha=0.3)
        ax_pdr.set_ylim([0, 105])
        
        # Delay chart
        ax_delay.plot(time, history['delay'], 'b-', linewidth=2)
        ax_delay.fill_between(time, history['delay'], alpha=0.3, color='blue')
        ax_delay.set_xlabel('Time (steps)', fontweight='bold')
        ax_delay.set_ylabel('Delay (ms)', fontweight='bold')
        ax_delay.set_title('Network Latency', fontsize=10, fontweight='bold')
        ax_delay.grid(True, alpha=0.3)
        
        fig.suptitle('Disaster-Resilient Mesh Network - Live Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig


# Example usage function
def visualize_simulation(simulator, history):
    """Quick function to visualize simulation results"""
    visualizer = MeshNetworkVisualizer(simulator)
    
    # Show current network state
    print("Displaying current network state...")
    visualizer.show_current_state()
    
    # Show metrics over time
    print("Displaying metrics over time...")
    visualizer.show_metrics_over_time(history)
    
    # Show interactive dashboard
    print("Displaying interactive dashboard...")
    visualizer.show_interactive_dashboard(history)
    
    plt.show()


if __name__ == "__main__":
    print("Mesh Network Visualizer Module")
    print("Import this module to use visualization in your simulation")
