# Powerball Insights
*Advanced Lottery Data Analysis & Machine Learning Platform*

An innovative Streamlit-powered application that combines sophisticated statistical analysis with machine learning predictions for Powerball lottery data. Built for data enthusiasts, researchers, and anyone curious about lottery number patterns.

## 🎯 Key Features

### Data Analysis & Visualization
- **Historical Analysis**: Comprehensive analysis of Powerball drawing history
- **Statistical Insights**: Frequency analysis, hot/cold numbers, and pattern detection
- **Interactive Visualizations**: Dynamic charts and graphs powered by Plotly and Altair
- **Combinatorial Analysis**: Deep dive into number pairs, triplets, and larger combinations

### Machine Learning Predictions
- **Ensemble Models**: Multiple ML algorithms including Random Forest, Gradient Boosting, and Ridge Regression
- **Feature Engineering**: Advanced pattern recognition using frequency, recency, and temporal features
- **Cross-Validation**: Time-series aware validation for reliable performance metrics
- **Prediction Tracking**: SQLite-based storage for prediction history and accuracy analysis

### Natural Language Interface
- **AI-Powered Queries**: Ask questions about lottery data in plain English
- **OpenAI Integration**: GPT-4o powered insights and analysis
- **Anthropic Support**: Claude-3.5-Sonnet for advanced pattern recognition
- **Smart Summaries**: Automated insights generation from complex datasets

### Data Management
- **Automated Maintenance**: Built-in data cleaning and validation tools
- **CSV Import/Export**: Flexible data format support with intelligent parsing
- **Date Standardization**: Consistent YYYY-MM-DD formatting across all systems
- **Backup Management**: Automated backup creation and retention policies

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Streamlit
- Required dependencies (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/RobbyMo81/LottoDataAnalyzer.git
   cd LottoDataAnalyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Optional: For AI-powered features
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Start exploring Powerball data insights!

## 📊 Application Modules

### Core Analytics
- **📈 Frequency Analysis**: Most and least drawn numbers with trend analysis
- **📅 Day of Week Analysis**: Drawing day patterns and performance metrics
- **🔄 Recency Analysis**: Recent appearance tracking and prediction weights
- **📊 Sum Analysis**: Number sum distributions and statistical patterns
- **🎲 Combinatorial Analysis**: Multi-number combination frequency and patterns

### Advanced Features
- **🤖 ML Experimental**: Real-time machine learning model training and evaluation
- **🎯 AutoML Tuning**: Automated hyperparameter optimization with experiment tracking
- **📝 Natural Language Queries**: AI-powered data exploration and insights
- **🛠️ Data Maintenance**: Comprehensive data cleaning and optimization tools
- **📁 CSV Formatter**: Intelligent data import with format detection and standardization

### Prediction Systems
- **Legacy Prediction Engine**: Multi-tool weighted prediction generation
- **Modern ML Pipeline**: SQLite-based prediction storage with enhanced features
- **Experimental Framework**: Cross-validation and performance benchmarking

## 🏗️ Architecture

### Backend Systems
- **SQLite Database**: Primary storage for predictions, models, and metadata
- **Pandas Pipeline**: Efficient data processing and transformation
- **Scikit-learn**: Machine learning model training and evaluation
- **Streamlit Framework**: Interactive web interface and user experience

### Storage Architecture
- **Unified SQLite**: Consolidated prediction and model storage
- **CSV Integration**: Flexible import/export capabilities
- **Backup System**: Automated data protection and recovery
- **Version Control**: Model and prediction versioning

### API Integration
- **OpenAI GPT-4o**: Advanced natural language processing
- **Anthropic Claude**: Pattern recognition and analysis
- **Web Scraping**: Real-time lottery data fetching capabilities

## 📋 System Requirements

### Core Dependencies
- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `plotly>=5.15.0` - Interactive visualizations
- `altair>=5.0.0` - Statistical visualization grammar

### AI & ML Extensions
- `openai>=1.0.0` - OpenAI API integration
- `anthropic>=0.25.0` - Anthropic API integration
- `joblib>=1.3.0` - Model serialization and parallel processing

### Data Processing
- `trafilatura>=1.6.0` - Web content extraction
- `matplotlib>=3.7.0` - Static plotting and visualization

## 🔧 Configuration

### Environment Setup
Create a `.streamlit/config.toml` file:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### API Keys (Optional)
For AI-powered features, configure environment variables:
- `OPENAI_API_KEY`: OpenAI API access for natural language queries
- `ANTHROPIC_API_KEY`: Anthropic API access for advanced analysis

## 📁 Project Structure

```
LottoDataAnalyzer/
├── app.py                    # Main Streamlit application
├── core/                     # Core application modules
│   ├── __init__.py          # Package initialization
│   ├── automl_demo.py       # AutoML tuning demonstrations
│   ├── combos.py            # Combinatorial analysis
│   ├── data_maintenance.py  # Data cleaning and validation
│   ├── datetime_manager.py  # Centralized date/time handling
│   ├── ml_experimental.py   # ML training and evaluation
│   └── prediction_system.py # Prediction generation engine
├── data/                    # Data storage directory
│   ├── *.csv               # Historical lottery data
│   ├── *.db                # SQLite databases
│   └── backups/            # Automatic backups
├── docs/                   # Documentation and reports
│   ├── *.md               # Architecture and analysis reports
│   └── diagrams/          # System diagrams and visualizations
└── requirements.txt       # Python dependencies
```

## 🤝 Contributing

We welcome contributions to improve Powerball Insights! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Implement your feature or bug fix
4. **Add tests**: Ensure your changes are well-tested
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**: Describe your changes and submit for review

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings for new functions and classes
- Include unit tests for new features
- Update documentation as needed

## 📝 Documentation

### Architecture Reports
- **Backend Architecture**: Complete system architecture documentation
- **Date/Time Analysis**: Centralized datetime handling analysis
- **ML Storage Analysis**: Machine learning storage and prediction systems
- **GitHub Integration**: Repository connection and workflow documentation

### API Documentation
- **Core Modules**: Detailed module-level documentation
- **Data Models**: Database schema and data structures
- **Prediction APIs**: Prediction generation and storage interfaces

## 🐛 Known Issues

### Current Limitations
- Prediction accuracy varies based on historical data quality
- Large datasets may require increased processing time
- AI features require external API keys for full functionality

### Troubleshooting
- **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
- **Database Issues**: Check SQLite file permissions in the `data/` directory
- **API Failures**: Verify API keys are correctly configured in environment variables

## 📈 Performance

### System Benchmarks
- **Data Loading**: <2 seconds for 5,000+ historical records
- **Prediction Generation**: <5 seconds for 5 predictions
- **ML Training**: 10-30 seconds for ensemble models
- **Cross-Validation**: 30-60 seconds for 5-fold validation

### Optimization Features
- **Lazy Loading**: On-demand data loading for improved startup times
- **Caching**: Streamlit caching for expensive computations
- **Efficient Storage**: SQLite indexing for fast query performance

## 🔒 Security & Privacy

### Data Protection
- **Local Storage**: All data processed locally, no external data transmission
- **API Security**: Secure handling of API keys through environment variables
- **Data Validation**: Input sanitization and validation throughout the pipeline

### Privacy Policy
- No personal data collection
- Lottery data analysis only
- API keys remain local to your environment

## 📞 Support

### Getting Help
- **Issues**: Open a GitHub issue for bug reports or feature requests
- **Documentation**: Check the `/docs` directory for detailed documentation
- **Community**: Join discussions in GitHub Discussions

### Contact
- **Repository**: [github.com/RobbyMo81/LottoDataAnalyzer](https://github.com/RobbyMo81/LottoDataAnalyzer)
- **Maintainer**: RobbyMo81

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit Team**: For the excellent web application framework
- **Scikit-learn Contributors**: For robust machine learning tools
- **Pandas Community**: For powerful data manipulation capabilities
- **OpenAI & Anthropic**: For AI-powered analysis capabilities

---

**Built with ❤️ for data enthusiasts and lottery researchers**

*Disclaimer: This application is for educational and research purposes only. Lottery games are games of chance, and past results do not guarantee future outcomes. Please gamble responsibly.*
