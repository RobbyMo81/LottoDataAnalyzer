"""
System Relationship Visualizer
-----------------------------
Creates SVG diagrams showing the relationship between ML Experimental and AutoML Tuning systems.
"""

import streamlit as st

def create_system_relationship_svg():
    """Generate SVG diagram showing system relationships."""
    
    svg_content = """
    <svg width="1000" height="700" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="mlExpGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#E8F4FD;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#B3D9FF;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="automlGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#FFF2E8;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#FFD9B3;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="sharedGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#E8F8E8;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#B3FFB3;stop-opacity:1" />
            </linearGradient>
        </defs>
        
        <!-- Title -->
        <text x="500" y="30" text-anchor="middle" font-size="24" font-weight="bold" fill="#333">
            ML Experimental ↔ AutoML Tuning System Architecture
        </text>
        
        <!-- ML Experimental System -->
        <rect x="50" y="80" width="350" height="280" rx="15" ry="15" 
              fill="url(#mlExpGradient)" stroke="#4A90E2" stroke-width="2"/>
        <text x="225" y="110" text-anchor="middle" font-size="18" font-weight="bold" fill="#2C5F8F">
            ML Experimental System
        </text>
        
        <!-- ML Experimental Components -->
        <text x="70" y="140" font-size="12" fill="#333">• Advanced Model Training Algorithms</text>
        <text x="70" y="160" font-size="12" fill="#333">• Memory-Optimized Processing</text>
        <text x="70" y="180" font-size="12" fill="#333">• Cross-Validation Framework</text>
        <text x="70" y="200" font-size="12" fill="#333">• Feature Engineering Pipeline</text>
        <text x="70" y="220" font-size="12" fill="#333">• Model Performance Tracking</text>
        <text x="70" y="240" font-size="12" fill="#333">• Ensemble Method Implementation</text>
        <text x="70" y="260" font-size="12" fill="#333">• Statistical Analysis Tools</text>
        <text x="70" y="280" font-size="12" fill="#333">• Data Preprocessing Utilities</text>
        <text x="70" y="300" font-size="12" fill="#333">• Bias Detection & Elimination</text>
        <text x="70" y="320" font-size="12" fill="#333">• Prediction Quality Assessment</text>
        
        <!-- AutoML Tuning System -->
        <rect x="600" y="80" width="350" height="280" rx="15" ry="15" 
              fill="url(#automlGradient)" stroke="#F5A623" stroke-width="2"/>
        <text x="775" y="110" text-anchor="middle" font-size="18" font-weight="bold" fill="#B8730E">
            AutoML Tuning System
        </text>
        
        <!-- AutoML Components -->
        <text x="620" y="140" font-size="12" fill="#333">• Hyperparameter Optimization</text>
        <text x="620" y="160" font-size="12" fill="#333">• Grid & Random Search Strategies</text>
        <text x="620" y="180" font-size="12" fill="#333">• Bayesian Optimization (Future)</text>
        <text x="620" y="200" font-size="12" fill="#333">• Experiment Tracking Database</text>
        <text x="620" y="220" font-size="12" fill="#333">• Configuration Management</text>
        <text x="620" y="240" font-size="12" fill="#333">• Trial Result Analysis</text>
        <text x="620" y="260" font-size="12" fill="#333">• Best Model Selection Logic</text>
        <text x="620" y="280" font-size="12" fill="#333">• Performance Comparison Tools</text>
        <text x="620" y="300" font-size="12" fill="#333">• Automated Parameter Tuning</text>
        <text x="620" y="320" font-size="12" fill="#333">• Convergence Analysis</text>
        
        <!-- Shared Infrastructure -->
        <rect x="250" y="400" width="500" height="120" rx="15" ry="15" 
              fill="url(#sharedGradient)" stroke="#7ED321" stroke-width="2"/>
        <text x="500" y="430" text-anchor="middle" font-size="16" font-weight="bold" fill="#5A9A17">
            Shared Infrastructure & Integration Layer
        </text>
        
        <text x="270" y="455" font-size="12" fill="#333">• SQLite Experiment Database with JSON Serialization</text>
        <text x="270" y="475" font-size="12" fill="#333">• Safe JSON Serialization Utilities (Boolean Handling)</text>
        <text x="270" y="495" font-size="12" fill="#333">• Data Storage & Versioning System</text>
        
        <!-- Integration Arrows -->
        <!-- ML Experimental to Shared -->
        <line x1="225" y1="360" x2="350" y2="400" stroke="#4A90E2" stroke-width="3" marker-end="url(#arrowblue)"/>
        <text x="280" y="380" font-size="10" fill="#4A90E2" font-weight="bold">Models & Results</text>
        
        <!-- AutoML to Shared -->
        <line x1="775" y1="360" x2="650" y2="400" stroke="#F5A623" stroke-width="3" marker-end="url(#arroworange)"/>
        <text x="700" y="380" font-size="10" fill="#F5A623" font-weight="bold">Experiments & Configs</text>
        
        <!-- Bidirectional Integration -->
        <line x1="400" y1="220" x2="600" y2="220" stroke="#7ED321" stroke-width="4" marker-end="url(#arrowgreen)" marker-start="url(#arrowgreen)"/>
        <text x="500" y="210" text-anchor="middle" font-size="12" fill="#7ED321" font-weight="bold">Live Integration</text>
        
        <!-- Arrow Markers -->
        <defs>
            <marker id="arrowblue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <polygon points="0,0 0,6 9,3" fill="#4A90E2"/>
            </marker>
            <marker id="arroworange" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <polygon points="0,0 0,6 9,3" fill="#F5A623"/>
            </marker>
            <marker id="arrowgreen" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                <polygon points="0,0 0,6 9,3" fill="#7ED321"/>
            </marker>
        </defs>
        
        <!-- Benefits Section -->
        <rect x="50" y="550" width="900" height="130" rx="10" ry="10" 
              fill="#F8F8F8" stroke="#CCC" stroke-width="1"/>
        <text x="500" y="575" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">
            Synergistic Benefits & Value Creation
        </text>
        
        <!-- Left Benefits -->
        <text x="70" y="600" font-size="14" font-weight="bold" fill="#4A90E2">ML Experimental → AutoML Tuning</text>
        <text x="90" y="620" font-size="11" fill="#333">• Provides robust, tested model training algorithms</text>
        <text x="90" y="635" font-size="11" fill="#333">• Supplies memory-optimized processing capabilities</text>
        <text x="90" y="650" font-size="11" fill="#333">• Offers advanced cross-validation and statistical techniques</text>
        <text x="90" y="665" font-size="11" fill="#333">• Delivers feature engineering insights and bias elimination</text>
        
        <!-- Right Benefits -->
        <text x="520" y="600" font-size="14" font-weight="bold" fill="#F5A623">AutoML Tuning → ML Experimental</text>
        <text x="540" y="620" font-size="11" fill="#333">• Automates hyperparameter optimization workflows</text>
        <text x="540" y="635" font-size="11" fill="#333">• Provides systematic experiment tracking and reproducibility</text>
        <text x="540" y="650" font-size="11" fill="#333">• Enables configuration management and version control</text>
        <text x="540" y="665" font-size="11" fill="#333">• Delivers performance comparison and optimization analytics</text>
    </svg>
    """
    
    return svg_content

def create_data_flow_diagram():
    """Generate SVG diagram showing data flow between systems."""
    
    svg_content = """
    <svg width="1000" height="600" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="processGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#E3F2FD;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#BBDEFB;stop-opacity:1" />
            </linearGradient>
            <marker id="flowArrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                <polygon points="0,0 0,6 9,3" fill="#2196F3"/>
            </marker>
        </defs>
        
        <!-- Title -->
        <text x="500" y="30" text-anchor="middle" font-size="22" font-weight="bold" fill="#333">
            Data Flow & Processing Pipeline
        </text>
        
        <!-- Process Flow Boxes -->
        <!-- Row 1: Data Input -->
        <rect x="50" y="70" width="120" height="60" rx="10" fill="url(#processGradient)" stroke="#2196F3" stroke-width="2"/>
        <text x="110" y="95" text-anchor="middle" font-size="11" font-weight="bold">Raw Powerball</text>
        <text x="110" y="110" text-anchor="middle" font-size="11" font-weight="bold">Data Ingestion</text>
        
        <rect x="220" y="70" width="120" height="60" rx="10" fill="url(#processGradient)" stroke="#2196F3" stroke-width="2"/>
        <text x="280" y="95" text-anchor="middle" font-size="11" font-weight="bold">Feature</text>
        <text x="280" y="110" text-anchor="middle" font-size="11" font-weight="bold">Engineering</text>
        
        <rect x="390" y="70" width="120" height="60" rx="10" fill="url(#processGradient)" stroke="#2196F3" stroke-width="2"/>
        <text x="450" y="95" text-anchor="middle" font-size="11" font-weight="bold">Data</text>
        <text x="450" y="110" text-anchor="middle" font-size="11" font-weight="bold">Preprocessing</text>
        
        <rect x="560" y="70" width="120" height="60" rx="10" fill="url(#processGradient)" stroke="#2196F3" stroke-width="2"/>
        <text x="620" y="95" text-anchor="middle" font-size="11" font-weight="bold">Train/Test</text>
        <text x="620" y="110" text-anchor="middle" font-size="11" font-weight="bold">Split</text>
        
        <!-- Row 2: ML Experimental Processing -->
        <rect x="50" y="200" width="120" height="60" rx="10" fill="#E8F4FD" stroke="#4A90E2" stroke-width="2"/>
        <text x="110" y="220" text-anchor="middle" font-size="11" font-weight="bold">Model</text>
        <text x="110" y="235" text-anchor="middle" font-size="11" font-weight="bold">Training</text>
        <text x="110" y="250" text-anchor="middle" font-size="10">(ML Experimental)</text>
        
        <rect x="220" y="200" width="120" height="60" rx="10" fill="#E8F4FD" stroke="#4A90E2" stroke-width="2"/>
        <text x="280" y="220" text-anchor="middle" font-size="11" font-weight="bold">Cross</text>
        <text x="280" y="235" text-anchor="middle" font-size="11" font-weight="bold">Validation</text>
        <text x="280" y="250" text-anchor="middle" font-size="10">(ML Experimental)</text>
        
        <rect x="390" y="200" width="120" height="60" rx="10" fill="#E8F4FD" stroke="#4A90E2" stroke-width="2"/>
        <text x="450" y="220" text-anchor="middle" font-size="11" font-weight="bold">Performance</text>
        <text x="450" y="235" text-anchor="middle" font-size="11" font-weight="bold">Evaluation</text>
        <text x="450" y="250" text-anchor="middle" font-size="10">(ML Experimental)</text>
        
        <!-- Row 3: AutoML Processing -->
        <rect x="50" y="330" width="120" height="60" rx="10" fill="#FFF2E8" stroke="#F5A623" stroke-width="2"/>
        <text x="110" y="350" text-anchor="middle" font-size="11" font-weight="bold">Parameter</text>
        <text x="110" y="365" text-anchor="middle" font-size="11" font-weight="bold">Search Space</text>
        <text x="110" y="380" text-anchor="middle" font-size="10">(AutoML)</text>
        
        <rect x="220" y="330" width="120" height="60" rx="10" fill="#FFF2E8" stroke="#F5A623" stroke-width="2"/>
        <text x="280" y="350" text-anchor="middle" font-size="11" font-weight="bold">Hyperparameter</text>
        <text x="280" y="365" text-anchor="middle" font-size="11" font-weight="bold">Optimization</text>
        <text x="280" y="380" text-anchor="middle" font-size="10">(AutoML)</text>
        
        <rect x="390" y="330" width="120" height="60" rx="10" fill="#FFF2E8" stroke="#F5A623" stroke-width="2"/>
        <text x="450" y="350" text-anchor="middle" font-size="11" font-weight="bold">Best Model</text>
        <text x="450" y="365" text-anchor="middle" font-size="11" font-weight="bold">Selection</text>
        <text x="450" y="380" text-anchor="middle" font-size="10">(AutoML)</text>
        
        <!-- Row 4: Shared Infrastructure -->
        <rect x="150" y="460" width="140" height="60" rx="10" fill="#E8F8E8" stroke="#7ED321" stroke-width="2"/>
        <text x="220" y="480" text-anchor="middle" font-size="11" font-weight="bold">Experiment</text>
        <text x="220" y="495" text-anchor="middle" font-size="11" font-weight="bold">Tracking</text>
        <text x="220" y="510" text-anchor="middle" font-size="10">(Shared)</text>
        
        <rect x="340" y="460" width="140" height="60" rx="10" fill="#E8F8E8" stroke="#7ED321" stroke-width="2"/>
        <text x="410" y="480" text-anchor="middle" font-size="11" font-weight="bold">Performance</text>
        <text x="410" y="495" text-anchor="middle" font-size="11" font-weight="bold">Analytics</text>
        <text x="410" y="510" text-anchor="middle" font-size="10">(Shared)</text>
        
        <rect x="530" y="460" width="140" height="60" rx="10" fill="#E8F8E8" stroke="#7ED321" stroke-width="2"/>
        <text x="600" y="480" text-anchor="middle" font-size="11" font-weight="bold">Model</text>
        <text x="600" y="495" text-anchor="middle" font-size="11" font-weight="bold">Deployment</text>
        <text x="600" y="510" text-anchor="middle" font-size="10">(Shared)</text>
        
        <!-- Flow Arrows -->
        <!-- Horizontal flows -->
        <line x1="170" y1="100" x2="210" y2="100" stroke="#2196F3" stroke-width="2" marker-end="url(#flowArrow)"/>
        <line x1="340" y1="100" x2="380" y2="100" stroke="#2196F3" stroke-width="2" marker-end="url(#flowArrow)"/>
        <line x1="510" y1="100" x2="550" y2="100" stroke="#2196F3" stroke-width="2" marker-end="url(#flowArrow)"/>
        
        <line x1="170" y1="230" x2="210" y2="230" stroke="#4A90E2" stroke-width="2" marker-end="url(#flowArrow)"/>
        <line x1="340" y1="230" x2="380" y2="230" stroke="#4A90E2" stroke-width="2" marker-end="url(#flowArrow)"/>
        
        <line x1="170" y1="360" x2="210" y2="360" stroke="#F5A623" stroke-width="2" marker-end="url(#flowArrow)"/>
        <line x1="340" y1="360" x2="380" y2="360" stroke="#F5A623" stroke-width="2" marker-end="url(#flowArrow)"/>
        
        <line x1="290" y1="490" x2="330" y2="490" stroke="#7ED321" stroke-width="2" marker-end="url(#flowArrow)"/>
        <line x1="480" y1="490" x2="520" y2="490" stroke="#7ED321" stroke-width="2" marker-end="url(#flowArrow)"/>
        
        <!-- Vertical flows -->
        <line x1="110" y1="130" x2="110" y2="190" stroke="#2196F3" stroke-width="2" marker-end="url(#flowArrow)"/>
        <line x1="280" y1="130" x2="280" y2="190" stroke="#2196F3" stroke-width="2" marker-end="url(#flowArrow)"/>
        <line x1="450" y1="130" x2="450" y2="190" stroke="#2196F3" stroke-width="2" marker-end="url(#flowArrow)"/>
        
        <line x1="110" y1="260" x2="110" y2="320" stroke="#4A90E2" stroke-width="2" marker-end="url(#flowArrow)"/>
        <line x1="280" y1="260" x2="280" y2="320" stroke="#4A90E2" stroke-width="2" marker-end="url(#flowArrow)"/>
        <line x1="450" y1="260" x2="450" y2="320" stroke="#4A90E2" stroke-width="2" marker-end="url(#flowArrow)"/>
        
        <line x1="220" y1="390" x2="220" y2="450" stroke="#F5A623" stroke-width="2" marker-end="url(#flowArrow)"/>
        <line x1="410" y1="390" x2="410" y2="450" stroke="#F5A623" stroke-width="2" marker-end="url(#flowArrow)"/>
        <line x1="600" y1="390" x2="600" y2="450" stroke="#F5A623" stroke-width="2" marker-end="url(#flowArrow)"/>
        
        <!-- Integration arrows between systems -->
        <line x1="560" y1="230" x2="620" y2="360" stroke="#7ED321" stroke-width="3" marker-end="url(#flowArrow)"/>
        <text x="590" y="280" font-size="10" fill="#7ED321" font-weight="bold">Integration</text>
        
        <!-- Legend -->
        <rect x="720" y="80" width="250" height="200" rx="10" fill="#F9F9F9" stroke="#DDD" stroke-width="1"/>
        <text x="845" y="105" text-anchor="middle" font-size="14" font-weight="bold">System Components</text>
        
        <rect x="730" y="120" width="15" height="15" fill="#E8F4FD" stroke="#4A90E2"/>
        <text x="755" y="132" font-size="11">ML Experimental</text>
        
        <rect x="730" y="145" width="15" height="15" fill="#FFF2E8" stroke="#F5A623"/>
        <text x="755" y="157" font-size="11">AutoML Tuning</text>
        
        <rect x="730" y="170" width="15" height="15" fill="#E8F8E8" stroke="#7ED321"/>
        <text x="755" y="182" font-size="11">Shared Infrastructure</text>
        
        <rect x="730" y="195" width="15" height="15" fill="url(#processGradient)" stroke="#2196F3"/>
        <text x="755" y="207" font-size="11">Data Processing</text>
        
        <text x="730" y="235" font-size="11" font-weight="bold">Key Features:</text>
        <text x="730" y="250" font-size="10">• Real-time integration</text>
        <text x="730" y="265" font-size="10">• Safe JSON serialization</text>
    </svg>
    """
    
    return svg_content

def render_text_based_architecture():
    """Render text-based system architecture when SVG fails."""
    st.markdown("""
    **System Architecture Overview (Text Format)**
    
    ```
    ┌─────────────────────────┐    ┌─────────────────────────┐
    │   ML Experimental       │◄──►│   AutoML Tuning         │
    │   System                │    │   System                │
    ├─────────────────────────┤    ├─────────────────────────┤
    │ • Model Training        │    │ • Hyperparameter Opt   │
    │ • Cross Validation      │    │ • Grid Search           │
    │ • Performance Tracking  │    │ • Experiment Tracking  │
    │ • Memory Management     │    │ • Best Model Selection │
    │ • Feature Engineering   │    │ • Configuration Mgmt   │
    └─────────────────────────┘    └─────────────────────────┘
                   │                           │
                   └───────────┬───────────────┘
                               │
                    ┌─────────────────────────┐
                    │  Shared Infrastructure  │
                    ├─────────────────────────┤
                    │ • SQLite Database       │
                    │ • JSON Serialization    │
                    │ • Data Storage          │
                    │ • Performance Analytics │
                    └─────────────────────────┘
    ```
    """)

def render_text_based_flow():
    """Render text-based data flow when SVG fails."""
    st.markdown("""
    **Data Flow Pipeline (Text Format)**
    
    ```
    Raw Data → Feature Engineering → Preprocessing → Train/Test Split
                                                           │
                                                           ▼
    ┌─────────────────┐                        ┌─────────────────┐
    │ ML Experimental │                        │ AutoML Tuning   │
    │                 │                        │                 │
    │ Model Training  │◄─────────────────────►│ Parameter Search │
    │ Cross Validation│                        │ Optimization    │
    │ Performance Eval│                        │ Best Selection  │
    └─────────────────┘                        └─────────────────┘
                │                                        │
                └────────────────┬───────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Result Integration    │
                    │                         │
                    │ • Experiment Tracking   │
                    │ • Performance Analytics │
                    │ • Model Deployment      │
                    └─────────────────────────┘
    ```
    """)

def render_page():
    """Render the complete system relationship visualization page."""
    
    st.header("🔗 System Architecture & Relationships")
    
    st.markdown("""
    This visualization shows how the ML Experimental and AutoML Tuning systems work together 
    to create a comprehensive machine learning platform for Powerball prediction analysis.
    """)
    
    # Main relationship diagram
    st.subheader("System Integration Architecture")
    
    # Use text-based rendering to avoid JavaScript errors
    render_text_based_architecture()
    
    # Data flow diagram
    st.subheader("Data Flow Pipeline")
    
    # Use text-based rendering to avoid JavaScript errors
    render_text_based_flow()
    
    # Detailed explanation
    st.subheader("How the Systems Benefit Each Other")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **ML Experimental → AutoML Tuning**
        
        The ML Experimental system provides the foundation that makes AutoML Tuning effective:
        
        - **Robust Algorithms**: Proven model training implementations that AutoML can optimize
        - **Memory Management**: Efficient processing that allows AutoML to run many trials
        - **Statistical Framework**: Advanced cross-validation and bias detection techniques
        - **Feature Engineering**: Domain-specific insights that improve AutoML search spaces
        - **Performance Metrics**: Comprehensive evaluation methods for comparing trials
        """)
    
    with col2:
        st.markdown("""
        **AutoML Tuning → ML Experimental**
        
        The AutoML Tuning system enhances ML Experimental capabilities:
        
        - **Automation**: Eliminates manual hyperparameter tuning across all models
        - **Systematic Search**: Organized exploration of parameter spaces
        - **Experiment Tracking**: Complete history of all trials and configurations
        - **Reproducibility**: Consistent results through configuration management
        - **Optimization**: Finds optimal parameters that improve prediction accuracy
        """)
    
    st.subheader("Technical Integration Points")
    
    st.markdown("""
    **Shared Infrastructure Components:**
    
    1. **Experiment Database**: SQLite database with safe JSON serialization for all experiment data
    2. **Configuration Management**: Unified system for storing and retrieving model parameters
    3. **Data Pipeline**: Common preprocessing and feature engineering workflows
    4. **Performance Analytics**: Shared metrics and evaluation frameworks
    
    **Critical Technical Solutions:**
    
    - **JSON Serialization**: Custom utilities handle complex data types including boolean parameters
    - **Memory Optimization**: Shared memory management prevents resource conflicts
    - **Real-time Integration**: Live data flow between systems during training and optimization
    - **Error Handling**: Comprehensive error management across both systems
    """)
    
    st.subheader("Business Value Creation")
    
    st.markdown("""
    **Combined System Benefits:**
    
    - **Accuracy Improvement**: AutoML finds optimal parameters for ML Experimental's advanced algorithms
    - **Development Speed**: Automated tuning accelerates model development cycles
    - **Reliability**: Systematic tracking ensures reproducible and auditable results
    - **Scalability**: Memory-optimized processing supports large-scale parameter searches
    - **User Experience**: Simplified interface hides complexity while providing powerful capabilities
    - **Continuous Learning**: System learns from each experiment to improve future predictions
    """)

if __name__ == "__main__":
    render_page()