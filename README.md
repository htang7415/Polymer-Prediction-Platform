# 🧪 Water-Soluble Polymer Discovery Platform

A comprehensive web application for exploring water-soluble polymer properties and making predictions using machine learning models.

## 🌟 Features

### 📊 Polymer Explorer
- **Interactive 3D Visualization**: Explore polymer properties in 3D space (χ, LogP, SA Score)
- **Dynamic Filtering**: Real-time filtering by property ranges
- **Multi-Class Support**: Polyester, polyether, polyimide, polyoxazolidone, polyurethane
- **Detailed Information**: Click points to view molecular structures and properties

### 🔮 Property Predictor
- **SMILES Input**: Predict properties from SMILES strings
- **ML-Powered**: Uses ensemble neural networks for accurate predictions
- **Multiple Properties**: χ (Chi parameter), LogP, SA Score, Tg (glass transition temperature)
- **Uncertainty Quantification**: Provides prediction confidence intervals
- **Molecular Visualization**: Automatic structure rendering from SMILES

## 🚀 Quick Start

### Local Development

1. **Clone and Setup**
   ```bash
   git clone <your-repo-url>
   cd polymer_app
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the App**
   - Open your browser to `http://localhost:8501`

### 🌐 Deploy to Streamlit Cloud

#### Prerequisites
- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

#### Step-by-Step Deployment

1. **Prepare Your Repository**
   ```bash
   # Initialize git repository
   git init
   git add .
   git commit -m "Initial commit: Water-soluble polymer discovery app"
   
   # Create GitHub repository and push
   git remote add origin https://github.com/yourusername/polymer-discovery.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account
   - Select your repository: `yourusername/polymer-discovery`
   - Set main file path: `app.py`
   - Click "Deploy!"

3. **Configuration**
   - Streamlit Cloud will automatically install dependencies from `requirements.txt`
   - The app will be available at: `https://your-app-name.streamlit.app`

#### Important Notes for Deployment

- **File Size Limits**: The sample data files are created to be under GitHub's 100MB limit
- **Model Files**: The PyTorch model files (`.pth`) are included but may need Git LFS for very large models
- **Dependencies**: All required packages are listed in `requirements.txt`

## 📁 Project Structure

```
polymer_app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── create_sample_data.py          # Script to generate sample data
├── data/                          # Sample polymer datasets
│   ├── polyester_with_SA_sample.csv
│   ├── polyether_with_SA_sample.csv
│   ├── polyimide_with_SA_sample.csv
│   ├── polyoxazolidone_with_SA_sample.csv
│   └── polyurethane_with_SA_sample.csv
├── models/                        # Pre-trained ML models
│   ├── Chi_model_light.pth        # Chi parameter prediction model (2MB)
│   └── Tg_model_light.pth         # Tg prediction model (6MB)
└── utils/                         # Utility modules
    ├── __init__.py
    ├── data_utils.py             # Data loading and processing
    ├── prediction_utils.py       # ML prediction utilities
    └── sascorer.py              # Synthetic accessibility scorer
```

## 📊 Dataset Information

### Polymer Classes
- **Polyester**: 50,000 samples
- **Polyether**: 33,359 samples  
- **Polyimide**: 50,000 samples
- **Polyoxazolidone**: 3,072 samples
- **Polyurethane**: 50,000 samples

### Properties
- **χ (Chi Parameter)**: Flory-Huggins interaction parameter
- **LogP**: Octanol-water partition coefficient
- **SA Score**: Synthetic accessibility score (1-10 scale)
- **Tg**: Glass transition temperature

## 🤖 Machine Learning Models

### Architecture
- **Chi Prediction**: SimpleNN with layers [256, 512, 128, 512]
- **Tg Prediction**: NeuralNetwork with layers [512, 1024, 512]
- **Lightweight**: Single model per property (GitHub-friendly file sizes)
- **Features**: 1024-bit Morgan fingerprints (radius=2)

### Performance
- Trained on large-scale polymer datasets
- Optimized for deployment with reduced file sizes
- Fast prediction times with single model inference

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **PyTorch**: Deep learning models
- **RDKit**: Chemical informatics and molecular visualization
- **Plotly**: Interactive 3D visualizations
- **Pandas/NumPy**: Data processing

### Browser Compatibility
- Chrome (recommended)
- Firefox
- Safari
- Edge

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure model files are in the `models/` directory
   - Check that PyTorch version is compatible

2. **Data Loading Issues**
   - Verify CSV files are in the `data/` directory
   - Run `create_sample_data.py` to regenerate sample files

3. **RDKit Installation**
   - Use conda for RDKit: `conda install -c conda-forge rdkit`
   - Or use pip: `pip install rdkit`

4. **Memory Issues**
   - Reduce sample sizes in data files
   - Use smaller batch sizes for predictions

### Performance Optimization
- Sample large datasets for visualization (10k points max)
- Use caching for model loading (`@st.cache_resource`)
- Enable GPU acceleration if available

## 📈 Future Enhancements

- [ ] Additional polymer classes
- [ ] More molecular descriptors
- [ ] Interactive structure editor
- [ ] Batch prediction upload
- [ ] Export functionality
- [ ] Advanced filtering options

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- RDKit for chemical informatics tools
- Streamlit for the web framework
- PyTorch for deep learning capabilities
- The polymer science community for datasets and insights

## 📞 Contact

For questions, issues, or collaborations, please contact:
- Email: [hao.tang@wisc.edu]
- GitHub: [htang7415]

---

**Happy Polymer Discovery! 🧪✨**
