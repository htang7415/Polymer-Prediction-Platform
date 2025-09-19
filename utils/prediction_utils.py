"""
Prediction utilities for polymer property prediction using ML models.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Handle NumPy import issues
try:
    import numpy as np
    # Force NumPy to use compatible version
    if hasattr(np, '__version__'):
        print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import error: {e}")
    sys.exit(1)

import torch
import torch.nn as nn
import pandas as pd
import streamlit as st
from typing import Dict, Optional, Tuple

# Handle RDKit imports with fallback
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors
    RDKIT_AVAILABLE = True
    
    # Debug: Print RDKit version info
    try:
        from rdkit import __version__ as rdkit_version
        print(f"RDKit version: {rdkit_version}")
    except ImportError:
        try:
            from rdkit import rdBase
            print(f"RDKit version: {rdBase.rdkitVersion}")
        except:
            print("RDKit version: Unknown")
            
    # Test Morgan API availability
    try:
        AllChem.GetMorganGenerator
        print("✅ GetMorganGenerator available (newer API)")
    except AttributeError:
        print("⚠️ GetMorganGenerator not available, will use fallback API")
        
except ImportError:
    print("Warning: RDKit not available. Some functionality will be limited.")
    RDKIT_AVAILABLE = False

# Import SA scorer
def calculateScore(mol):
    """Default SA score calculation if sascorer fails"""
    return 5.0

try:
    from .sascorer import calculateScore
except ImportError:
    try:
        import sys
        import os
        # Add current directory to path for sascorer import
        current_dir = os.path.dirname(__file__)
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        from sascorer import calculateScore
    except ImportError:
        # Keep default function if all imports fail
        pass

class SimpleNN(nn.Module):
    """Chi parameter prediction model architecture"""
    def __init__(self, input_dim, layers, dropout=0.0):
        super().__init__()
        net = []
        prev = input_dim
        for h in layers:
            net.append(nn.Linear(prev, h))
            net.append(nn.ReLU())
            if dropout > 0:
                net.append(nn.Dropout(dropout))
            prev = h
        net.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        return self.net(x)

class NeuralNetwork(nn.Module):
    """Tg prediction model architecture"""
    def __init__(self, n_input, *neurons):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_neurons = n_input
        for n in neurons:
            layers.extend([nn.Linear(prev_neurons, n), nn.ReLU()])
            prev_neurons = n
        layers.append(nn.Linear(prev_neurons, 1))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class PolymerPredictor:
    """Main class for polymer property prediction"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chi_model = None
        self.tg_model = None
        self.models_loaded = False
        
    def load_models(self):
        """Load the lightweight models for prediction"""
        if self.models_loaded:
            return
            
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        # Load Chi model (single lightweight model)
        chi_model_path = os.path.join(models_dir, 'Chi_model_light.pth')
        if os.path.exists(chi_model_path):
            try:
                chi_state_dict = torch.load(chi_model_path, map_location=self.device)
                chi_layers = [256, 512, 128, 512]
                chi_dropout = 0.07
                self.chi_model = SimpleNN(1024, chi_layers, dropout=chi_dropout).to(self.device)
                self.chi_model.load_state_dict(chi_state_dict)
                print("✅ Chi model loaded successfully!")
            except Exception as e:
                print(f"❌ Could not load Chi model: {e}")
                self.chi_model = None
        else:
            print("❌ Chi model file not found - Chi predictions will be unavailable")
            self.chi_model = None
        
        # Load Tg model (single lightweight model)
        tg_model_path = os.path.join(models_dir, 'Tg_model_light.pth')
        if os.path.exists(tg_model_path):
            try:
                tg_state_dict = torch.load(tg_model_path, map_location=self.device)
                tg_neurons = (512, 1024, 512)
                self.tg_model = NeuralNetwork(1024, *tg_neurons).to(self.device)
                self.tg_model.load_state_dict(tg_state_dict)
                print("✅ Tg model loaded successfully!")
            except Exception as e:
                print(f"❌ Could not load Tg model: {e}")
                self.tg_model = None
        else:
            print("❌ Tg model file not found - Tg predictions will be unavailable")
            self.tg_model = None
        
        self.models_loaded = True
    
    def canonical_smiles(self, smiles: str) -> str:
        """Convert SMILES to canonical form"""
        if not RDKIT_AVAILABLE:
            return smiles  # Return as-is if RDKit not available
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            raise ValueError(f"Invalid SMILES: {smiles}")
    
    def smiles_to_morgan(self, smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
        """Convert SMILES to Morgan fingerprint"""
        if not RDKIT_AVAILABLE:
            return np.zeros(n_bits, dtype=np.float32)  # Return zero vector if RDKit not available
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        
        # Use the older, more compatible Morgan fingerprint API
        # Force cache refresh - updated 2025-09-19
        try:
            # Try the newer API first (RDKit 2023+)
            morgan_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
            fingerprint = morgan_gen.GetFingerprint(mol)
            return np.array(fingerprint, dtype=np.float32)
        except AttributeError:
            # Fall back to older API (RDKit <2023) - more compatible with rdkit-pypi
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            return np.array(fingerprint, dtype=np.float32)
    
    def calculate_logp(self, smiles: str) -> float:
        """Calculate LogP using RDKit"""
        if not RDKIT_AVAILABLE:
            return 0.0  # Default value if RDKit not available
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.nan
            return round(Crippen.MolLogP(mol), 3)
        except:
            return np.nan
    
    def calculate_sa_score(self, smiles: str) -> float:
        """Calculate synthetic accessibility score"""
        if not RDKIT_AVAILABLE:
            return 3.5  # Default moderate complexity if RDKit not available
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.nan
            
            # Try to calculate SA score using the original sascorer
            try:
                score = calculateScore(mol)
                if score is not None and score != 5.0:
                    return round(score, 3)
            except:
                pass
            
            # Fallback: improved molecular complexity estimation
            if RDKIT_AVAILABLE and mol is not None:
                num_atoms = mol.GetNumAtoms()
                num_rings = mol.GetRingInfo().NumRings()
                num_rotatable = Descriptors.NumRotatableBonds(mol)
                num_aromatic = len([atom for atom in mol.GetAtoms() if atom.GetIsAromatic()])
                num_hetero = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6]])
            else:
                # Default values if RDKit not available
                num_atoms = 20
                num_rings = 1
                num_rotatable = 3
                num_aromatic = 6
                num_hetero = 2
            
            # Simplified and more conservative SA score calculation
            # Base score for simple molecules
            base_score = 2.0  # Start higher for typical organic molecules
            
            # Size contribution (less aggressive)
            if num_atoms <= 10:
                size_contribution = 0.0  # Very small molecules are easy
            elif num_atoms <= 20:
                size_contribution = 0.5  # Small molecules
            elif num_atoms <= 30:
                size_contribution = 1.0  # Medium molecules
            else:
                size_contribution = 1.5  # Larger molecules
            
            # Ring contribution (moderate)
            ring_contribution = num_rings * 0.3
            
            # Rotatable bonds (small contribution)
            rotatable_contribution = min(num_rotatable * 0.1, 0.5)
            
            # Aromatic bonus (aromatic rings are generally easier)
            aromatic_bonus = -min(num_aromatic * 0.2, 1.0)
            
            # Heteroatom contribution (small)
            hetero_contribution = min(num_hetero * 0.2, 0.8)
            
            score = base_score + size_contribution + ring_contribution + rotatable_contribution + aromatic_bonus + hetero_contribution
            
            # Keep score in reasonable range (typical SA scores are 1-6)
            score = min(max(score, 1.0), 6.0)
            
            return round(score, 3)
            
        except Exception as e:
            # Simple fallback
            return 3.5  # Moderate complexity default
    
    def predict_chi(self, smiles: str) -> Tuple[float, float]:
        """Predict Chi parameter (single model, no uncertainty)"""
        if self.chi_model is None:
            return float('nan'), float('nan')
        
        try:
            canonical_smi = self.canonical_smiles(smiles)
            fingerprint = self.smiles_to_morgan(canonical_smi)
            
            # Ensure fingerprint is the right type and shape
            if len(fingerprint) != 1024:
                print(f"Warning: Fingerprint length {len(fingerprint)} != 1024")
                return float('nan'), float('nan')
                
            # Convert to PyTorch tensor with explicit type handling
            try:
                X_tensor = torch.tensor(fingerprint, dtype=torch.float32, device=self.device).unsqueeze(0)
            except Exception as tensor_e:
                print(f"Tensor conversion error: {tensor_e}")
                return float('nan'), float('nan')
            
            self.chi_model.eval()
            with torch.no_grad():
                try:
                    prediction = self.chi_model(X_tensor).cpu().detach().numpy().item()
                except Exception as pred_e:
                    print(f"Model prediction error: {pred_e}")
                    return float('nan'), float('nan')
            
            # Check for NaN prediction
            if prediction != prediction:  # NaN check without numpy
                print("Warning: Chi model returned NaN")
                return float('nan'), float('nan')
            
            # Return prediction with zero uncertainty (single model)
            return round(float(prediction), 3), 0.0
            
        except Exception as e:
            print(f"Error in Chi prediction: {e}")
            st.error(f"Error in Chi prediction: {e}")
            return float('nan'), float('nan')
    
    def predict_tg(self, smiles: str) -> Tuple[float, float]:
        """Predict Tg (single model, no uncertainty)"""
        if self.tg_model is None:
            return float('nan'), float('nan')
        
        try:
            fingerprint = self.smiles_to_morgan(smiles)
            
            # Ensure fingerprint is the right type and shape
            if len(fingerprint) != 1024:
                print(f"Warning: Fingerprint length {len(fingerprint)} != 1024")
                return float('nan'), float('nan')
                
            # Convert to PyTorch tensor with explicit type handling
            try:
                X_tensor = torch.tensor(fingerprint, dtype=torch.float32, device=self.device).unsqueeze(0)
            except Exception as tensor_e:
                print(f"Tensor conversion error: {tensor_e}")
                return float('nan'), float('nan')
            
            self.tg_model.eval()
            with torch.no_grad():
                try:
                    prediction = self.tg_model(X_tensor).cpu().detach().numpy().item()
                except Exception as pred_e:
                    print(f"Model prediction error: {pred_e}")
                    return float('nan'), float('nan')
            
            # Check for NaN prediction
            if prediction != prediction:  # NaN check without numpy
                print("Warning: Tg model returned NaN")
                return float('nan'), float('nan')
            
            # Return prediction with zero uncertainty (single model)
            return round(float(prediction), 3), 0.0
            
        except Exception as e:
            print(f"Error in Tg prediction: {e}")
            st.error(f"Error in Tg prediction: {e}")
            return float('nan'), float('nan')
    
    def predict_properties(self, smiles: str) -> Dict[str, float]:
        """
        Predict all properties for a given SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with predicted properties
        """
        # Load models if not already loaded
        self.load_models()
        
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        results = {}
        
        # Predict Chi parameter
        chi_mean, chi_std = self.predict_chi(smiles)
        results['chi_mean'] = chi_mean
        results['chi_std'] = chi_std
        
        # Predict Tg
        tg_mean, tg_std = self.predict_tg(smiles)
        results['tg_mean'] = tg_mean
        results['tg_std'] = tg_std
        
        # Calculate LogP
        results['logp'] = self.calculate_logp(smiles)
        
        # Calculate SA Score
        results['sa_score'] = self.calculate_sa_score(smiles)
        
        return results
    
    def batch_predict(self, smiles_list: list) -> pd.DataFrame:
        """
        Predict properties for a batch of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            DataFrame with predictions
        """
        self.load_models()
        
        results = []
        for smiles in smiles_list:
            try:
                props = self.predict_properties(smiles)
                props['smiles'] = smiles
                results.append(props)
            except Exception as e:
                st.warning(f"Could not predict for {smiles}: {e}")
        
        return pd.DataFrame(results)

@st.cache_resource
def get_predictor():
    """Get cached predictor instance - version 2 with Morgan API fix"""
    return PolymerPredictor()
