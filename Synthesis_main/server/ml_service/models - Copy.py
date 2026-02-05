import torch
import torch.nn as nn
from config import *

# ==================== ATTENTION MECHANISM ====================
class Attention(nn.Module):
    """Attention mechanism for focusing on important features."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return x * weights

# ==================== TABULAR GAN (CWGAN-GP) ====================
class TabularGenerator(nn.Module):
    """Conditional Wasserstein GAN Generator for Tabular Data."""
    def __init__(self):
        super().__init__()
        input_dim = LATENT_DIM + len(COND_FEATURES)
        output_dim = len([f for f in TABULAR_FEATURES if f != 'hypertension']) + 2
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        return self.model(x)

class TabularDiscriminator(nn.Module):
    """Conditional Wasserstein GAN Discriminator for Tabular Data."""
    def __init__(self):
        super().__init__()
        input_dim = len([f for f in TABULAR_FEATURES if f != 'hypertension']) + 2 + len(COND_FEATURES)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        return self.model(x)

# ==================== TIME SERIES GAN ====================
class TimeSeriesGenerator(nn.Module):
    """Conditional GAN Generator for Time Series (RBS) with LSTM."""
    def __init__(self):
        super().__init__()
        input_dim = LATENT_DIM + len(COND_FEATURES)
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, SEQ_LENGTH * HIDDEN_DIM),
            nn.LayerNorm(SEQ_LENGTH * HIDDEN_DIM),
            nn.LeakyReLU(0.2)
        )
        
        self.lstm = nn.LSTM(
            input_size=HIDDEN_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.attn = Attention(HIDDEN_DIM)
        
        self.out = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.LSTM)):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z, cond):
        x = torch.cat([z, cond], dim=1)
        x = self.fc(x)
        x = x.view(-1, SEQ_LENGTH, HIDDEN_DIM)
        x, _ = self.lstm(x)
        x = self.attn(x)
        x = self.out(x)
        return x

class TimeSeriesDiscriminator(nn.Module):
    """Conditional Wasserstein GAN Discriminator for Time Series."""
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM + len(COND_FEATURES), 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.LSTM)):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, cond):
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = torch.cat([x, cond], dim=1)
        return self.fc(x)

# ==================== CROSS-MODAL GENERATOR ====================
class CrossModalGenerator(nn.Module):
    """Cross-modal generator for translating between tabular and time series."""
    def __init__(self):
        super().__init__()
        tabular_size = len([f for f in TABULAR_FEATURES if f != 'hypertension']) + 2
        
        self.tab_to_ts = nn.Sequential(
            nn.Linear(tabular_size + len(COND_FEATURES), 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, SEQ_LENGTH * HIDDEN_DIM),
            nn.LayerNorm(SEQ_LENGTH * HIDDEN_DIM),
            nn.LeakyReLU(0.2)
        )
        
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        
        self.ts_out = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.ts_encoder = nn.LSTM(1, HIDDEN_DIM, num_layers=2, batch_first=True)
        
        self.ts_to_tab = nn.Sequential(
            nn.Linear(HIDDEN_DIM + len(COND_FEATURES), 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, tabular_size),
            nn.Sigmoid()
        )
    
    def generate_ts_from_tab(self, tab, cond):
        """Generate time series from tabular data."""
        x = torch.cat([tab, cond], dim=1)
        x = self.tab_to_ts(x)
        x = x.view(-1, SEQ_LENGTH, HIDDEN_DIM)
        x, _ = self.lstm(x)
        x = self.ts_out(x)
        return x
    
    def generate_tab_from_ts(self, ts, cond):
        """Generate tabular data from time series."""
        _, (h_n, _) = self.ts_encoder(ts)
        x = h_n[-1]
        x = torch.cat([x, cond], dim=1)
        return self.ts_to_tab(x)
