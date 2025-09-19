"""
Water-Soluble Polymer Discovery Platform
========================================

A Streamlit web application for exploring water-soluble polymer properties and making predictions.

Features:
1. Interactive 3D visualization of polymer properties
2. SMILES-based property prediction using ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import base64
from io import BytesIO
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Handle RDKit imports with fallback for deployment issues
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Crippen
    RDKIT_AVAILABLE = True
    
    # Try to import drawing functionality
    try:
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
        RDKIT_DRAW_AVAILABLE = True
    except ImportError as e:
        st.warning("RDKit drawing functionality not available. Molecular structures will not be displayed.")
        RDKIT_DRAW_AVAILABLE = False
        
except ImportError as e:
    st.error("RDKit not available. Some functionality will be limited.")
    RDKIT_AVAILABLE = False
    RDKIT_DRAW_AVAILABLE = False

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import custom modules
try:
    from prediction_utils import PolymerPredictor, get_predictor
    from data_utils import load_polymer_data, create_sample_data
except ImportError as e:
    st.error(f"Error importing utilities: {e}")

# Page configuration
st.set_page_config(
    page_title="Water-Soluble Polymer Discovery",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 5px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def render_mol_structure(smiles, size=(300, 300)):
    """Render molecular structure from SMILES"""
    if not RDKIT_AVAILABLE or not RDKIT_DRAW_AVAILABLE:
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        img_data = drawer.GetDrawingText()
        img_base64 = base64.b64encode(img_data).decode()
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        # Don't show error in deployment, just return None
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🧪 Water-Soluble Polymer Discovery Platform</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a feature:",
        ["🏠 Home", "📊 Polymer Explorer", "🔮 Property Predictor"]
    )
    
    if page == "🏠 Home":
        render_home_page()
    elif page == "📊 Polymer Explorer":
        render_explorer_page()
    elif page == "🔮 Property Predictor":
        render_predictor_page()

def render_home_page():
    """Render the home page"""
    st.markdown('<h2 class="sub-header">Welcome to the Polymer Discovery Platform</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>🔍 Polymer Explorer</h3>
        <p>Explore our extensive database of water-soluble polymers with interactive 3D visualizations. 
        Filter by properties and discover polymer structures that meet your specific requirements.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Features:
        - **Interactive 3D Plots**: Visualize χ (chi), LogP, and SA Score
        - **Dynamic Filtering**: Adjust property ranges in real-time
        - **Detailed Information**: Click points to see structures and properties
        - **Multiple Polymer Classes**: Polyester, polyether, polyimide, polyoxazolidone, polyurethane
        """)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>🤖 Property Predictor</h3>
        <p>Use our state-of-the-art machine learning models to predict polymer properties from SMILES strings. 
        Get instant predictions with uncertainty estimates.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Predicted Properties:
        - **χ (Chi Parameter)**: Flory-Huggins interaction parameter
        - **Tg**: Glass transition temperature
        - **LogP**: Octanol-water partition coefficient  
        - **SA Score**: Synthetic accessibility score
        - **Molecular Structure**: Visual representation from SMILES
        """)
    
    # Statistics section
    st.markdown('<h2 class="sub-header">📈 Database Statistics</h2>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>250K+</h3>
        <p>Total Polymers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>5</h3>
        <p>Polymer Classes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>2</h3>
        <p>ML Models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3>3</h3>
        <p>Properties</p>
        </div>
        """, unsafe_allow_html=True)

def render_explorer_page():
    """Render the polymer explorer page"""
    st.markdown('<h2 class="sub-header">📊 Polymer Explorer</h2>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading polymer data..."):
        try:
            data = load_polymer_data()
            if data.empty:
                st.error("No data available. Please check data files.")
                return
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    # Sidebar filters
    st.sidebar.markdown("### 🎛️ Filters")
    
    # Polymer class filter
    polymer_classes = data['polymer_class'].unique() if 'polymer_class' in data.columns else []
    selected_classes = st.sidebar.multiselect(
        "Select Polymer Classes:",
        options=polymer_classes,
        default=polymer_classes[:3] if len(polymer_classes) > 3 else polymer_classes
    )
    
    # Property range filters
    if not data.empty and len(selected_classes) > 0:
        filtered_data = data[data['polymer_class'].isin(selected_classes)]
        
        # Chi filter
        if 'chi_mean' in filtered_data.columns:
            chi_min, chi_max = float(filtered_data['chi_mean'].min()), float(filtered_data['chi_mean'].max())
            chi_range = st.sidebar.slider(
                "χ (Chi) Range:",
                min_value=chi_min,
                max_value=chi_max,
                value=(chi_min, chi_max),
                step=0.01
            )
        else:
            chi_range = (0, 1)
        
        # LogP filter
        if 'LogP' in filtered_data.columns:
            logp_min, logp_max = float(filtered_data['LogP'].min()), float(filtered_data['LogP'].max())
            logp_range = st.sidebar.slider(
                "LogP Range:",
                min_value=logp_min,
                max_value=logp_max,
                value=(logp_min, logp_max),
                step=0.1
            )
        else:
            logp_range = (-5, 5)
        
        # SA Score filter
        if 'SA_Score' in filtered_data.columns:
            sa_min, sa_max = float(filtered_data['SA_Score'].min()), float(filtered_data['SA_Score'].max())
            sa_range = st.sidebar.slider(
                "SA Score Range:",
                min_value=sa_min,
                max_value=sa_max,
                value=(sa_min, sa_max),
                step=0.1
            )
        else:
            sa_range = (1, 10)
        
        # Apply filters
        mask = (
            (filtered_data['chi_mean'] >= chi_range[0]) & (filtered_data['chi_mean'] <= chi_range[1]) &
            (filtered_data['LogP'] >= logp_range[0]) & (filtered_data['LogP'] <= logp_range[1]) &
            (filtered_data['SA_Score'] >= sa_range[0]) & (filtered_data['SA_Score'] <= sa_range[1])
        )
        plot_data = filtered_data[mask]
        
        # Display statistics with larger fonts
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
            <h2 style="margin: 0; font-size: 2rem;">{len(plot_data):,}</h2>
            <h4 style="margin: 0;">Total Polymers</h4>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
            <h2 style="margin: 0; font-size: 2rem;">{len(selected_classes)}</h2>
            <h4 style="margin: 0;">Selected Classes</h4>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
            <h2 style="margin: 0; font-size: 2rem;">{len(plot_data):,}/{len(filtered_data):,}</h2>
            <h4 style="margin: 0;">Filtered Results</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # 3D Scatter Plot
        if len(plot_data) > 0:
            # Sample data for performance but keep index for selection
            sample_data = plot_data.sample(min(10000, len(plot_data))).reset_index(drop=True)
            
            # Create hover data with all relevant information
            hover_cols = ['polym', 'mon1', 'mon2', 'Tg_mean'] if all(col in sample_data.columns for col in ['polym', 'mon1', 'mon2', 'Tg_mean']) else ['polym']
            
            fig = px.scatter_3d(
                sample_data,
                x='chi_mean',
                y='LogP', 
                z='SA_Score',
                color='polymer_class',
                hover_data=hover_cols,
                title="3D Visualization of Polymer Properties",
                labels={
                    'chi_mean': 'χ (Chi Parameter)',
                    'LogP': 'LogP (Octanol-Water Partition)',
                    'SA_Score': 'SA Score (Synthetic Accessibility)',
                    'Tg_mean': 'Tg (Glass Transition Temperature)'
                }
            )
            
            fig.update_layout(
                height=700,
                scene=dict(
                    xaxis_title='χ (Chi Parameter)',
                    yaxis_title='LogP',
                    zaxis_title='SA Score'
                ),
                legend=dict(
                    font=dict(size=14)  # Larger legend font
                )
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Handle point selection with manual selection
            st.markdown("### 🔍 Polymer Details")
            
            # Add a selectbox for manual polymer selection
            st.markdown("**Select a polymer to view details:**")
            
            # Create a selection interface
            col_select1, col_select2 = st.columns([1, 2])
            
            with col_select1:
                selected_class = st.selectbox(
                    "Polymer Class:",
                    options=sample_data['polymer_class'].unique(),
                    key="polymer_class_select"
                )
            
            with col_select2:
                # Filter by selected class
                class_data = sample_data[sample_data['polymer_class'] == selected_class]
                polymer_options = [f"Polymer {i+1}" for i in range(min(10, len(class_data)))]
                selected_idx = st.selectbox(
                    "Select Polymer:",
                    options=range(len(polymer_options)),
                    format_func=lambda x: polymer_options[x],
                    key="polymer_select"
                )
            
            if selected_class and selected_idx is not None:
                class_data = sample_data[sample_data['polymer_class'] == selected_class]
                if len(class_data) > selected_idx:
                    selected_polymer = class_data.iloc[selected_idx]
                    
                    # Display detailed information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🧪 Polymer Properties**")
                        st.write(f"**Class:** {selected_polymer.get('polymer_class', 'N/A')}")
                        st.write(f"**χ (Chi):** {selected_polymer.get('chi_mean', 'N/A'):.3f}")
                        st.write(f"**LogP:** {selected_polymer.get('LogP', 'N/A'):.3f}")
                        st.write(f"**SA Score:** {selected_polymer.get('SA_Score', 'N/A'):.3f}")
                        if 'Tg_mean' in selected_polymer:
                            st.write(f"**Tg:** {selected_polymer.get('Tg_mean', 'N/A'):.1f} K")
                    
                    with col2:
                        st.markdown("**🧬 Molecular Information**")
                        if 'polym' in selected_polymer:
                            st.write(f"**Polymer SMILES:** `{selected_polymer['polym']}`")
                        if 'mon1' in selected_polymer:
                            st.write(f"**Monomer 1:** `{selected_polymer['mon1']}`")
                        if 'mon2' in selected_polymer and selected_polymer['mon2'] is not None and str(selected_polymer['mon2']) != 'nan':
                            st.write(f"**Monomer 2:** `{selected_polymer['mon2']}`")
                    
                    # Try to show molecular structure
                    if 'polym' in selected_polymer:
                        try:
                            img_data = render_mol_structure(selected_polymer['polym'], size=(400, 300))
                            if img_data:
                                st.markdown("**🔬 Molecular Structure**")
                                st.markdown(
                                    f'<div style="text-align: center;"><img src="{img_data}" style="max-width: 400px;"></div>',
                                    unsafe_allow_html=True
                                )
                        except:
                            pass
            else:
                st.info("Select a polymer class and polymer above to see detailed information.")
            
            # Display sample data table with Tg
            st.markdown("### 📋 Sample Data")
            display_cols = ['polymer_class', 'chi_mean', 'LogP', 'SA_Score']
            if 'Tg_mean' in plot_data.columns:
                display_cols.append('Tg_mean')
            if 'polym' in plot_data.columns:
                display_cols.append('polym')
            if 'mon1' in plot_data.columns:
                display_cols.append('mon1')
            if 'mon2' in plot_data.columns:
                display_cols.append('mon2')
            
            st.dataframe(
                plot_data[display_cols].head(100),
                use_container_width=True
            )
        
        else:
            st.warning("No polymers match the selected criteria. Please adjust your filters.")
    
    else:
        st.warning("Please select at least one polymer class to begin exploration.")

def render_predictor_page():
    """Render the property predictor page"""
    st.markdown('<h2 class="sub-header">🔮 Property Predictor</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p>Enter a SMILES string below to predict polymer properties using our trained machine learning models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for selected example
    if 'selected_example_smiles' not in st.session_state:
        st.session_state.selected_example_smiles = ""
    
    # Example SMILES
    st.markdown("### 💡 Example Polymer SMILES")
    st.markdown("Click on any example below to automatically predict its properties:")
    
    examples = [
        ("Polyester", "*OC(CC(*)=O)CC(C)=O"),
        ("Polyether", "*OC1CC(C=C)CCC1*"),
        ("Polyimide", "*C1CC(=O)CCC1N1C(=O)CN(CCN2CC(=O)N(*)C(=O)C2)CC1=O"),
        ("Polyurethane", "*Oc1ccc(Cl)cc1Cc1cc(Cl)ccc1OC(=O)Nc1cccc(NC(*)=O)c1C")
    ]
    
    cols = st.columns(len(examples))
    selected_example = None
    
    for i, (name, smiles) in enumerate(examples):
        with cols[i]:
            # Show shortened SMILES in button but use full SMILES
            short_smiles = smiles[:20] + "..." if len(smiles) > 20 else smiles
            if st.button(f"🧪 {name}\n`{short_smiles}`", key=f"example_{i}", use_container_width=True):
                st.session_state.selected_example_smiles = smiles
                selected_example = (name, smiles)
                st.rerun()
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Use selected example or allow manual input
        default_value = st.session_state.selected_example_smiles
        smiles_input = st.text_input(
            "Enter Polymer SMILES string:",
            value=default_value,
            placeholder="e.g., *OC(CC(*)=O)CC(C)=O (or click examples above)",
            help="Enter a valid SMILES representation of your polymer. Use * to denote connection points."
        )
    
    with col2:
        predict_button = st.button("🚀 Predict Properties", type="primary")
    
    # Auto-predict if example was just selected
    auto_predict = (st.session_state.selected_example_smiles != "" and 
                   smiles_input == st.session_state.selected_example_smiles and 
                   smiles_input != "")
    
    # Clear the selected example after use
    if auto_predict:
        st.session_state.selected_example_smiles = ""
    
    # Prediction logic
    if (predict_button and smiles_input) or auto_predict:
        if auto_predict:
            st.success("🎯 Auto-predicting properties for selected example...")
        with st.spinner("Predicting properties..."):
            try:
                # Validate SMILES
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(smiles_input)
                    if mol is None:
                        st.error("❌ Invalid SMILES string. Please check your input.")
                        return
                else:
                    st.warning("⚠️ RDKit not available - SMILES validation disabled")
                    mol = None
                
                # Display molecular structure
                st.markdown("### 🧬 Molecular Structure")
                img_data = render_mol_structure(smiles_input)
                if img_data:
                    st.markdown(
                        f'<div style="text-align: center;"><img src="{img_data}" style="max-width: 400px;"></div>',
                        unsafe_allow_html=True
                    )
                
                # Make predictions
                st.markdown("### 📊 Predicted Properties")
                
                # Initialize predictor (cached)
                predictor = get_predictor()
                
                # Predict properties
                results = predictor.predict_properties(smiles_input)
                
                # Display results in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    chi_uncertainty = results.get('chi_std', 0)
                    chi_display = f"{results.get('chi_mean', 0):.3f}"
                    if chi_uncertainty > 0:
                        chi_display += f" ± {chi_uncertainty:.3f}"
                    
                    st.markdown("""
                    <div class="metric-card">
                    <h3>χ (Chi Parameter)</h3>
                    <h2>{}</h2>
                    <p>Flory-Huggins interaction parameter</p>
                    </div>
                    """.format(chi_display), 
                    unsafe_allow_html=True)
                
                with col2:
                    tg_uncertainty = results.get('tg_std', 0)
                    tg_kelvin = results.get('tg_mean', 0)
                    tg_celsius = tg_kelvin - 273.15 if tg_kelvin else 0
                    tg_display = f"{tg_celsius:.1f}"
                    if tg_uncertainty > 0:
                        tg_display += f" ± {tg_uncertainty:.1f}"
                    
                    st.markdown("""
                    <div class="metric-card">
                    <h3>Tg (°C)</h3>
                    <h2>{}</h2>
                    <p>Glass transition temperature</p>
                    </div>
                    """.format(tg_display), 
                    unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="metric-card">
                    <h3>LogP</h3>
                    <h2>{:.3f}</h2>
                    <p>Octanol-water partition coefficient</p>
                    </div>
                    """.format(results.get('logp', 0)), 
                    unsafe_allow_html=True)
                
                with col4:
                    st.markdown("""
                    <div class="metric-card">
                    <h3>SA Score</h3>
                    <h2>{:.3f}</h2>
                    <p>Synthetic accessibility score</p>
                    </div>
                    """.format(results.get('sa_score', 0)), 
                    unsafe_allow_html=True)
                
                # Additional molecular properties
                st.markdown("### 🔬 Additional Properties")
                
                # Calculate additional RDKit descriptors
                if RDKIT_AVAILABLE and mol is not None:
                    mw = Descriptors.MolWt(mol)
                    tpsa = Descriptors.TPSA(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Molecular Weight", f"{mw:.2f} g/mol")
                    with col2:
                        st.metric("TPSA", f"{tpsa:.2f} Ų")
                    with col3:
                        st.metric("H-Bond Donors", int(hbd))
                    with col4:
                        st.metric("H-Bond Acceptors", int(hba))
                else:
                    st.info("Additional molecular properties require RDKit")
                
                # Property interpretation
                st.markdown("### 📝 Property Interpretation")
                
                interpretation = []
                
                chi_val = results.get('chi_mean', 0)
                if chi_val < 0.5:
                    interpretation.append("✅ **Low χ parameter**: Good polymer-solvent compatibility")
                else:
                    interpretation.append("⚠️ **High χ parameter**: Poor polymer-solvent compatibility")
                
                tg_val = results.get('tg_mean', 0) - 273.15  # Convert to Celsius
                if tg_val < 0:  # Below 0°C
                    interpretation.append("❄️ **Low Tg**: Flexible polymer at room temperature")
                elif tg_val < 100:  # Below 100°C
                    interpretation.append("🌡️ **Medium Tg**: Semi-rigid polymer at room temperature")
                else:
                    interpretation.append("🔥 **High Tg**: Rigid polymer at room temperature")
                
                logp_val = results.get('logp', 0)
                if logp_val < 0:
                    interpretation.append("✅ **Negative LogP**: Hydrophilic, water-soluble")
                else:
                    interpretation.append("⚠️ **Positive LogP**: Lipophilic, less water-soluble")
                
                sa_val = results.get('sa_score', 0)
                if sa_val < 3:
                    interpretation.append("✅ **Low SA Score**: Easy to synthesize")
                elif sa_val < 5:
                    interpretation.append("⚠️ **Medium SA Score**: Moderately difficult to synthesize")
                else:
                    interpretation.append("❌ **High SA Score**: Difficult to synthesize")
                
                for item in interpretation:
                    st.markdown(item)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please check that all model files are available and try again.")
    
    elif predict_button:
        st.warning("Please enter a SMILES string first.")

if __name__ == "__main__":
    main()
