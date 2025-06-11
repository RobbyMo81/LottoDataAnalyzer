"""
ML System Relationship Visualization
-----------------------------------
Creates a comprehensive diagram showing the relationship between
ML Experimental and AutoML Tuning systems.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import numpy as np

def create_system_relationship_diagram():
    """Create a detailed diagram showing ML system relationships."""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 15)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'ml_experimental': '#E8F4FD',
        'automl_tuning': '#FFF2E8',
        'shared': '#E8F8E8',
        'data_flow': '#4A90E2',
        'benefit': '#F5A623',
        'integration': '#7ED321'
    }
    
    # Title
    ax.text(10, 14.5, 'ML Experimental ↔ AutoML Tuning System Relationship', 
            fontsize=20, fontweight='bold', ha='center')
    
    # ML Experimental System (Left Side)
    ml_exp_box = FancyBboxPatch((0.5, 8), 8, 5.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['ml_experimental'],
                                edgecolor='#4A90E2', linewidth=2)
    ax.add_patch(ml_exp_box)
    
    ax.text(4.5, 12.8, 'ML Experimental System', fontsize=16, fontweight='bold', ha='center')
    
    # ML Experimental Components
    ml_components = [
        'Advanced Model Training',
        'Memory-Optimized Processing',
        'Cross-Validation Framework',
        'Feature Engineering Pipeline',
        'Model Performance Tracking',
        'Ensemble Methods'
    ]
    
    for i, component in enumerate(ml_components):
        ax.text(1, 12.2 - i*0.6, f'• {component}', fontsize=11, va='center')
    
    # AutoML Tuning System (Right Side)
    automl_box = FancyBboxPatch((11.5, 8), 8, 5.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['automl_tuning'],
                                edgecolor='#F5A623', linewidth=2)
    ax.add_patch(automl_box)
    
    ax.text(15.5, 12.8, 'AutoML Tuning System', fontsize=16, fontweight='bold', ha='center')
    
    # AutoML Components
    automl_components = [
        'Hyperparameter Optimization',
        'Grid & Random Search',
        'Experiment Tracking',
        'Configuration Management',
        'Trial Result Analysis',
        'Best Model Selection'
    ]
    
    for i, component in enumerate(automl_components):
        ax.text(12, 12.2 - i*0.6, f'• {component}', fontsize=11, va='center')
    
    # Shared Infrastructure (Center)
    shared_box = FancyBboxPatch((7, 5.5), 6, 2,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['shared'],
                                edgecolor='#7ED321', linewidth=2)
    ax.add_patch(shared_box)
    
    ax.text(10, 7, 'Shared Infrastructure', fontsize=14, fontweight='bold', ha='center')
    
    shared_components = [
        '• Experiment Tracker Database',
        '• JSON Serialization Utilities',
        '• Data Storage & Versioning'
    ]
    
    for i, component in enumerate(shared_components):
        ax.text(7.2, 6.4 - i*0.3, component, fontsize=10, va='center')
    
    # Data Flow Arrows
    # ML Experimental to Shared
    arrow1 = patches.FancyArrowPatch((8.5, 10), (9, 7.5),
                                     arrowstyle='->', mutation_scale=20,
                                     color=colors['data_flow'], linewidth=2)
    ax.add_patch(arrow1)
    
    # AutoML to Shared
    arrow2 = patches.FancyArrowPatch((11.5, 10), (11, 7.5),
                                     arrowstyle='->', mutation_scale=20,
                                     color=colors['data_flow'], linewidth=2)
    ax.add_patch(arrow2)
    
    # Bidirectional between systems
    arrow3 = patches.FancyArrowPatch((8.5, 11), (11.5, 11),
                                     arrowstyle='<->', mutation_scale=20,
                                     color=colors['integration'], linewidth=3)
    ax.add_patch(arrow3)
    
    # Benefits Section (Bottom)
    benefits_box = FancyBboxPatch((1, 1), 18, 3.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#F8F8F8',
                                  edgecolor='#666666', linewidth=1)
    ax.add_patch(benefits_box)
    
    ax.text(10, 4, 'Synergistic Benefits', fontsize=16, fontweight='bold', ha='center')
    
    # Left Benefits (ML Experimental Benefits AutoML)
    ax.text(2, 3.4, 'ML Experimental → AutoML Tuning', fontsize=12, fontweight='bold', color='#4A90E2')
    ml_benefits = [
        '• Provides robust model training algorithms',
        '• Supplies memory-optimized processing capabilities',
        '• Offers advanced cross-validation techniques',
        '• Delivers feature engineering insights'
    ]
    
    for i, benefit in enumerate(ml_benefits):
        ax.text(2.2, 3 - i*0.25, benefit, fontsize=10)
    
    # Right Benefits (AutoML Benefits ML Experimental)
    ax.text(11, 3.4, 'AutoML Tuning → ML Experimental', fontsize=12, fontweight='bold', color='#F5A623')
    automl_benefits = [
        '• Automates hyperparameter optimization',
        '• Provides systematic experiment tracking',
        '• Enables configuration reproducibility',
        '• Delivers performance comparison analytics'
    ]
    
    for i, benefit in enumerate(automl_benefits):
        ax.text(11.2, 3 - i*0.25, benefit, fontsize=10)
    
    # Integration Points
    integration_points = [
        (4.5, 8.5, 'Model\nSelection'),
        (15.5, 8.5, 'Parameter\nOptimization'),
        (10, 5, 'Experiment\nLogging')
    ]
    
    for x, y, label in integration_points:
        circle = Circle((x, y), 0.4, facecolor=colors['integration'], 
                       edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, fontsize=9, ha='center', va='center', 
                fontweight='bold', color='white')
    
    # Legend
    legend_elements = [
        patches.Patch(color=colors['ml_experimental'], label='ML Experimental System'),
        patches.Patch(color=colors['automl_tuning'], label='AutoML Tuning System'),
        patches.Patch(color=colors['shared'], label='Shared Infrastructure'),
        patches.Patch(color=colors['integration'], label='Integration Points')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.15))
    
    plt.tight_layout()
    return fig

def create_data_flow_diagram():
    """Create a detailed data flow diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Data Flow & Processing Pipeline', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Process boxes
    processes = [
        (2, 8, 'Raw Data\nIngestion'),
        (6, 8, 'Feature\nEngineering'),
        (10, 8, 'Model\nTraining'),
        (2, 5.5, 'Hyperparameter\nSearch Space'),
        (6, 5.5, 'AutoML\nOptimization'),
        (10, 5.5, 'Best Model\nSelection'),
        (2, 3, 'Experiment\nTracking'),
        (6, 3, 'Performance\nAnalysis'),
        (10, 3, 'Model\nDeployment')
    ]
    
    for x, y, label in processes:
        box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor='lightblue',
                            edgecolor='navy', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, label, fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Flow arrows
    flows = [
        ((2.8, 8), (5.2, 8)),
        ((6.8, 8), (9.2, 8)),
        ((2.8, 5.5), (5.2, 5.5)),
        ((6.8, 5.5), (9.2, 5.5)),
        ((2.8, 3), (5.2, 3)),
        ((6.8, 3), (9.2, 3)),
        ((2, 7.6), (2, 5.9)),
        ((6, 7.6), (6, 5.9)),
        ((10, 7.6), (10, 5.9)),
        ((2, 5.1), (2, 3.4)),
        ((6, 5.1), (6, 3.4)),
        ((10, 5.1), (10, 3.4))
    ]
    
    for start, end in flows:
        arrow = patches.FancyArrowPatch(start, end,
                                       arrowstyle='->', mutation_scale=15,
                                       color='darkblue', linewidth=2)
        ax.add_patch(arrow)
    
    # Add system labels
    ax.text(1, 9, 'ML Experimental Pipeline', fontsize=14, fontweight='bold', 
            color='#4A90E2', rotation=90, va='center')
    ax.text(1, 6.5, 'AutoML Tuning Pipeline', fontsize=14, fontweight='bold', 
            color='#F5A623', rotation=90, va='center')
    ax.text(1, 4, 'Shared Analytics Pipeline', fontsize=14, fontweight='bold', 
            color='#7ED321', rotation=90, va='center')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create and save the relationship diagram
    fig1 = create_system_relationship_diagram()
    fig1.savefig('ml_system_relationship.png', dpi=300, bbox_inches='tight')
    
    # Create and save the data flow diagram
    fig2 = create_data_flow_diagram()
    fig2.savefig('ml_data_flow.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("✓ ML System relationship diagrams created successfully")
    print("✓ Saved as: ml_system_relationship.png and ml_data_flow.png")