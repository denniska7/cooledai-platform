#!/usr/bin/env python3
"""
Recurrent Physics-Informed Neural Network (Recurrent PINN)

This module implements a recurrent neural network that combines LSTM-based
temporal modeling with physics-informed constraints for predicting temperature
evolution during data center cooling failures.

Architecture:
    Input: [T_current, Q_load, u_flow] at time t
    LSTM Backbone: Captures temporal dependencies
    Output Heads: Multi-horizon predictions [T_{t+1}, T_{t+5}, T_{t+10}]
    Physics Loss: Enforces dT/dt = Q/(m·c_p) constraint

Key Features:
    - Multi-horizon temperature prediction (1, 5, 10 seconds ahead)
    - Physics-informed loss function
    - Time-to-Failure estimation
    - Handles multiple failure modes
    - Recurrent architecture for sequential data

Author: CoolingAI Simulator
Date: 2026-01-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

# Physics constants
RHO_AIR = 1.184  # kg/m³
C_P_AIR = 1005.0  # J/(kg·K)
ROOM_VOLUME = 240.0  # m³
THERMAL_MASS_MULTIPLIER = 3.0  # Account for equipment thermal inertia
THERMAL_MASS = RHO_AIR * ROOM_VOLUME * THERMAL_MASS_MULTIPLIER  # kg
THERMAL_CAPACITY = THERMAL_MASS * C_P_AIR  # J/K

# Critical thresholds
T_WARNING = 65.0    # °C
T_CRITICAL = 85.0   # °C
T_SHUTDOWN = 95.0   # °C
T_DAMAGE = 100.0    # °C

# Normalization constants (for stable training)
T_MEAN = 30.0       # °C
T_STD = 20.0        # °C
Q_MEAN = 100000.0   # W (100 kW)
Q_STD = 50000.0     # W (50 kW)
U_MEAN = 1.75       # m/s
U_STD = 1.0         # m/s


# ============================================================================
# MC DROPOUT FOR UNCERTAINTY QUANTIFICATION
# ============================================================================

class MCDropout(nn.Module):
    """
    Monte Carlo Dropout layer that remains active during inference.

    Standard nn.Dropout turns off during eval() mode, but MC Dropout stays active
    to provide uncertainty estimates via multiple forward passes with different
    dropout masks.

    Usage:
        During training: Behaves like standard dropout
        During inference: Run model multiple times to get prediction distribution
            predictions = [model(x) for _ in range(n_samples)]
            mean = torch.mean(torch.stack(predictions), dim=0)
            std = torch.std(torch.stack(predictions), dim=0)
            confidence_95 = 1.96 * std  # 95% confidence interval
    """

    def __init__(self, p: float = 0.1):
        """
        Initialize MC Dropout.

        Args:
            p: Dropout probability
        """
        super(MCDropout, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout that remains active during inference.

        Args:
            x: Input tensor

        Returns:
            Output with dropout applied (always, even in eval mode)
        """
        # Always apply dropout, regardless of training mode
        return F.dropout(x, p=self.p, training=True)


# ============================================================================
# SELF-ATTENTION MECHANISM (STEP 3.3)
# ============================================================================

class TemporalSelfAttention(nn.Module):
    """
    Self-Attention mechanism for temporal sequence modeling.

    Allows the model to weight which previous time-steps are most important
    for predicting the current thermal state. This is crucial for capturing
    thermal spikes and transient events.

    Architecture:
        Query, Key, Value projections → Scaled Dot-Product Attention → Output

    Args:
        hidden_dim: Dimension of LSTM hidden state
        num_heads: Number of attention heads (default: 4)
        dropout: Attention dropout probability
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super(TemporalSelfAttention, self).__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out = nn.Linear(hidden_dim, hidden_dim)

        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention to input sequence.

        Args:
            x: Input tensor, shape [batch, seq_len, hidden_dim]
            mask: Optional attention mask, shape [batch, seq_len, seq_len]

        Returns:
            output: Attention output, shape [batch, seq_len, hidden_dim]
            attn_weights: Attention weights, shape [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, hidden_dim = x.size()

        # Residual connection
        residual = x

        # Linear projections: [batch, seq_len, hidden_dim] → [batch, seq_len, hidden_dim]
        Q = self.query(x)  # Query: "What am I looking for?"
        K = self.key(x)    # Key: "What do I contain?"
        V = self.value(x)  # Value: "What information do I have?"

        # Reshape for multi-head attention
        # [batch, seq_len, hidden_dim] → [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, num_heads, seq_len, seq_len]

        # Apply mask if provided (for causal attention or padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]

        # Concatenate heads
        # [batch, num_heads, seq_len, head_dim] → [batch, seq_len, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)

        # Output projection
        output = self.out(attn_output)
        output = self.out_dropout(output)

        # Residual connection + Layer normalization
        output = self.norm(output + residual)

        return output, attn_weights


# ============================================================================
# RECURRENT PINN ARCHITECTURE
# ============================================================================

class RecurrentPINN(nn.Module):
    """
    Recurrent Physics-Informed Neural Network for temperature prediction.

    Combines LSTM for temporal modeling with physics-informed constraints
    to predict multi-horizon temperature trajectories during cooling failures.

    Architecture:
        Input Layer → LSTM Layers → Dense Layers → Multi-Horizon Heads

    Input Features (3):
        - T_current: Current temperature (°C)
        - Q_load: IT equipment heat load (W)
        - u_flow: Air flow velocity (m/s)

    Output Predictions (3):
        - T_{t+1}: Temperature 1 second ahead
        - T_{t+5}: Temperature 5 seconds ahead
        - T_{t+10}: Temperature 10 seconds ahead

    Physics Constraints:
        - dT/dt = Q / (m · c_p)
        - Energy conservation
        - Physical bounds on temperature
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        num_dense_layers: int = 3,
        dropout: float = 0.1,
        use_physics_constraints: bool = True,
        use_thermal_inertia: bool = True,
        use_mc_dropout: bool = True,
        use_attention: bool = True,
        num_attention_heads: int = 4
    ):
        """
        Initialize Recurrent PINN with Thermal Inertia Modeling + Self-Attention.

        Args:
            input_dim: Number of input features (default: 3)
            hidden_dim: Hidden dimension for LSTM and dense layers
            num_lstm_layers: Number of LSTM layers (default: 2)
            num_dense_layers: Number of dense layers after LSTM
            dropout: Dropout probability for regularization
            use_physics_constraints: Whether to apply physics constraints
            use_thermal_inertia: Enable learnable thermal mass and hysteresis (Step 3.2)
            use_mc_dropout: Enable MC Dropout for uncertainty quantification
            use_attention: Enable self-attention on LSTM output (Step 3.3)
            num_attention_heads: Number of attention heads (Step 3.3)
        """
        super(RecurrentPINN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.use_physics_constraints = use_physics_constraints
        self.use_thermal_inertia = use_thermal_inertia
        self.use_mc_dropout = use_mc_dropout
        self.use_attention = use_attention
        self.dropout_p = dropout

        # Input normalization
        self.register_buffer('T_mean', torch.tensor(T_MEAN))
        self.register_buffer('T_std', torch.tensor(T_STD))
        self.register_buffer('Q_mean', torch.tensor(Q_MEAN))
        self.register_buffer('Q_std', torch.tensor(Q_STD))
        self.register_buffer('U_mean', torch.tensor(U_MEAN))
        self.register_buffer('U_std', torch.tensor(U_STD))

        # ====================================================================
        # STEP 3.2: THERMAL INERTIA MODELING
        # ====================================================================

        if use_thermal_inertia:
            # Learnable Thermal Mass Parameter
            # Represents hardware-specific heat capacity: C_hardware = m_hw * c_p_hw
            # Initialized to THERMAL_CAPACITY (856,742 J/K), will be learned during training
            # This allows model to adapt to different rack configurations
            self.log_thermal_mass = nn.Parameter(
                torch.tensor(np.log(THERMAL_MASS))  # ~6.75
            )

            # Residual Heat State (for hysteresis logic)
            # Tracks heat energy that hasn't dissipated yet from previous time steps
            # Shape will be [batch, hidden_dim] - stored in LSTM hidden state
            self.residual_heat_projection = nn.Linear(hidden_dim, hidden_dim)

            # Cooling rate limiter (prevents unrealistic temperature drops)
            # Learned parameter that limits how fast temperature can decrease
            self.log_max_cooling_rate = nn.Parameter(
                torch.tensor(np.log(0.5))  # Initialize to 0.5°C/s max cooling
            )

        # LSTM backbone for temporal modeling with hysteresis
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=False  # Causal (only past → future)
        )

        # ====================================================================
        # STEP 3.3: TEMPORAL SELF-ATTENTION
        # ====================================================================

        if use_attention:
            # Self-Attention layer to weight which previous time-steps are most important
            # This helps the model focus on critical thermal events (e.g., spikes, failures)
            self.attention = TemporalSelfAttention(
                hidden_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            )

        # Dense layers with MC Dropout support
        dense_layers = []
        current_dim = hidden_dim
        for i in range(num_dense_layers):
            dense_layers.append(nn.Linear(current_dim, hidden_dim))
            dense_layers.append(nn.LayerNorm(hidden_dim))
            dense_layers.append(nn.ReLU())
            # MC Dropout: Use custom dropout that stays active during inference
            if use_mc_dropout:
                dense_layers.append(MCDropout(dropout))
            else:
                dense_layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        self.dense = nn.Sequential(*dense_layers)

        # Multi-horizon prediction heads
        # Each head predicts temperature at a specific future time
        self.head_t1 = nn.Linear(hidden_dim, 1)   # t+1 second
        self.head_t5 = nn.Linear(hidden_dim, 1)   # t+5 seconds
        self.head_t10 = nn.Linear(hidden_dim, 1)  # t+10 seconds

        # Physics-based auxiliary head (predicts dT/dt for physics loss)
        self.physics_head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights
                    nn.init.xavier_uniform_(param)
                elif 'linear' in name or 'head' in name:
                    # Linear layer weights
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def normalize_input(
        self,
        T: torch.Tensor,
        Q: torch.Tensor,
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize input features for stable training.

        Args:
            T: Temperature (°C), shape [batch, seq_len] or [batch]
            Q: Heat load (W), shape [batch, seq_len] or [batch]
            u: Flow velocity (m/s), shape [batch, seq_len] or [batch]

        Returns:
            Normalized input tensor, shape [batch, seq_len, 3] or [batch, 3]
        """
        T_norm = (T - self.T_mean) / self.T_std
        Q_norm = (Q - self.Q_mean) / self.Q_std
        u_norm = (u - self.U_mean) / self.U_std

        # Handle different input shapes
        if T.dim() == 1:
            # Single time step: [batch] → [batch, 1, 3]
            return torch.stack([T_norm, Q_norm, u_norm], dim=-1).unsqueeze(1)
        elif T.dim() == 2:
            # Sequence: [batch, seq_len] → [batch, seq_len, 3]
            return torch.stack([T_norm, Q_norm, u_norm], dim=-1)
        else:
            raise ValueError(f"Invalid input shape: {T.shape}")

    def denormalize_temperature(self, T_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize temperature predictions."""
        return T_norm * self.T_std + self.T_mean

    def forward(
        self,
        T_current: torch.Tensor,
        Q_load: torch.Tensor,
        u_flow: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Recurrent PINN with Thermal Inertia Modeling.

        Args:
            T_current: Current temperature (°C), shape [batch] or [batch, seq_len]
            Q_load: IT heat load (W), shape [batch] or [batch, seq_len]
            u_flow: Air flow velocity (m/s), shape [batch] or [batch, seq_len]
            hidden_state: Optional LSTM hidden state for sequential inference

        Returns:
            Dictionary with keys:
                - 'T_t1': Temperature at t+1 second, shape [batch, 1]
                - 'T_t5': Temperature at t+5 seconds, shape [batch, 1]
                - 'T_t10': Temperature at t+10 seconds, shape [batch, 1]
                - 'dT_dt': Predicted heating rate (°C/s), shape [batch, 1]
                - 'hidden': LSTM hidden state for next step
                - 'thermal_mass': Learned thermal mass (if thermal_inertia enabled)
                - 'max_cooling_rate': Learned max cooling rate (if thermal_inertia enabled)
        """
        # Normalize inputs
        x = self.normalize_input(T_current, Q_load, u_flow)
        # x shape: [batch, seq_len, 3] or [batch, 1, 3]

        # LSTM encoding
        if hidden_state is not None:
            lstm_out, hidden = self.lstm(x, hidden_state)
        else:
            lstm_out, hidden = self.lstm(x)
        # lstm_out shape: [batch, seq_len, hidden_dim]

        # ====================================================================
        # STEP 3.3: TEMPORAL SELF-ATTENTION
        # ====================================================================
        if self.use_attention:
            # Apply self-attention to weight important time-steps
            # This helps the model focus on critical thermal events
            lstm_out, attn_weights = self.attention(lstm_out)
            # lstm_out shape: [batch, seq_len, hidden_dim] (after attention)
            # attn_weights shape: [batch, num_heads, seq_len, seq_len]
        else:
            attn_weights = None

        # ====================================================================
        # STEP 3.2: HYSTERESIS LOGIC - Residual Heat Carryover
        # ====================================================================
        if self.use_thermal_inertia and hidden_state is not None:
            # Extract cell state (c) which carries long-term information
            h, c = hidden

            # Project residual heat from previous time step
            # Cell state encodes accumulated thermal energy that hasn't dissipated
            residual_heat = self.residual_heat_projection(c[-1])  # Last LSTM layer

            # Blend current LSTM output with residual heat
            # This prevents temperature from dropping faster than physically possible
            lstm_out_last = lstm_out[:, -1, :] if lstm_out.size(1) > 1 else lstm_out.squeeze(1)
            blended_features = lstm_out_last + 0.3 * residual_heat  # 30% residual contribution
        else:
            # Take last time step for prediction (standard behavior)
            if lstm_out.size(1) > 1:
                blended_features = lstm_out[:, -1, :]  # [batch, hidden_dim]
            else:
                blended_features = lstm_out.squeeze(1)  # [batch, hidden_dim]

        # Dense feature extraction
        features = self.dense(blended_features)  # [batch, hidden_dim]

        # Multi-horizon predictions (normalized)
        T_t1_norm = self.head_t1(features)   # [batch, 1]
        T_t5_norm = self.head_t5(features)   # [batch, 1]
        T_t10_norm = self.head_t10(features) # [batch, 1]

        # Denormalize predictions
        T_t1 = self.denormalize_temperature(T_t1_norm)
        T_t5 = self.denormalize_temperature(T_t5_norm)
        T_t10 = self.denormalize_temperature(T_t10_norm)

        # ====================================================================
        # STEP 3.2: COOLING RATE LIMITER - Prevent Unrealistic Drops
        # ====================================================================
        if self.use_thermal_inertia:
            # Get learned thermal parameters
            thermal_mass = torch.exp(self.log_thermal_mass)
            max_cooling_rate = torch.exp(self.log_max_cooling_rate)

            # Handle both scalar and batch tensor inputs
            if T_current.dim() == 1:
                T_curr = T_current
            else:
                T_curr = T_current[:, -1] if T_current.size(1) > 1 else T_current.squeeze(1)

            # Apply cooling rate limiter to prevent unrealistic temperature drops
            # When load decreases, temperature shouldn't drop faster than max_cooling_rate
            T_t1_clamped = torch.maximum(
                T_t1.squeeze(),
                T_curr - max_cooling_rate * 1.0  # 1 second timestep
            ).unsqueeze(1)

            T_t5_clamped = torch.maximum(
                T_t5.squeeze(),
                T_curr - max_cooling_rate * 5.0  # 5 second timestep
            ).unsqueeze(1)

            T_t10_clamped = torch.maximum(
                T_t10.squeeze(),
                T_curr - max_cooling_rate * 10.0  # 10 second timestep
            ).unsqueeze(1)

            # Use clamped predictions
            T_t1, T_t5, T_t10 = T_t1_clamped, T_t5_clamped, T_t10_clamped

        # Physics-based heating rate prediction
        dT_dt = self.physics_head(features)  # [batch, 1]

        # Prepare output dictionary
        output = {
            'T_t1': T_t1,
            'T_t5': T_t5,
            'T_t10': T_t10,
            'dT_dt': dT_dt,
            'hidden': hidden,
            'features': features  # For auxiliary tasks
        }

        # Add thermal inertia parameters if enabled
        if self.use_thermal_inertia:
            output['thermal_mass'] = torch.exp(self.log_thermal_mass)
            output['max_cooling_rate'] = torch.exp(self.log_max_cooling_rate)

        # Add attention weights if enabled (Step 3.3)
        if self.use_attention and attn_weights is not None:
            output['attn_weights'] = attn_weights  # [batch, num_heads, seq_len, seq_len]

        return output

    def predict_sequence(
        self,
        T_initial: torch.Tensor,
        Q_load: torch.Tensor,
        u_flow: torch.Tensor,
        num_steps: int = 60
    ) -> torch.Tensor:
        """
        Predict temperature sequence autoregressively.

        Uses t+1 predictions recursively to generate long sequences.

        Args:
            T_initial: Initial temperature, shape [batch]
            Q_load: IT heat load (constant or time-varying), shape [batch] or [batch, num_steps]
            u_flow: Air flow velocity, shape [batch] or [batch, num_steps]
            num_steps: Number of time steps to predict

        Returns:
            Temperature trajectory, shape [batch, num_steps]
        """
        batch_size = T_initial.size(0)
        device = T_initial.device

        # Handle constant vs time-varying Q and u
        if Q_load.dim() == 1:
            Q_load = Q_load.unsqueeze(1).expand(-1, num_steps)
        if u_flow.dim() == 1:
            u_flow = u_flow.unsqueeze(1).expand(-1, num_steps)

        # Initialize
        T_sequence = torch.zeros(batch_size, num_steps, device=device)
        T_current = T_initial
        hidden = None

        # Autoregressive prediction
        for t in range(num_steps):
            # Forward pass
            outputs = self.forward(
                T_current,
                Q_load[:, t],
                u_flow[:, t],
                hidden_state=hidden
            )

            # Use t+1 prediction as next input
            T_next = outputs['T_t1'].squeeze(1)
            T_sequence[:, t] = T_next

            # Update for next iteration
            T_current = T_next
            hidden = outputs['hidden']

        return T_sequence

    def estimate_time_to_failure(
        self,
        T_current: torch.Tensor,
        Q_load: torch.Tensor,
        u_flow: torch.Tensor,
        threshold: float = T_CRITICAL,
        max_time: int = 300
    ) -> torch.Tensor:
        """
        Estimate time until temperature reaches critical threshold.

        Args:
            T_current: Current temperature, shape [batch]
            Q_load: IT heat load, shape [batch]
            u_flow: Air flow velocity, shape [batch]
            threshold: Critical temperature threshold (default: 85°C)
            max_time: Maximum time to simulate (seconds)

        Returns:
            Time-to-Failure (seconds), shape [batch]
            Returns max_time if threshold not reached
        """
        batch_size = T_current.size(0)
        device = T_current.device

        # Predict temperature sequence
        T_sequence = self.predict_sequence(
            T_current, Q_load, u_flow, num_steps=max_time
        )

        # Find first time when threshold is exceeded
        exceeds_threshold = T_sequence >= threshold  # [batch, max_time]

        # Time-to-Failure for each sample
        ttf = torch.full((batch_size,), float(max_time), device=device)

        for i in range(batch_size):
            if exceeds_threshold[i].any():
                ttf[i] = exceeds_threshold[i].nonzero(as_tuple=True)[0][0].float()

        return ttf

    def predict_with_uncertainty(
        self,
        T_current: torch.Tensor,
        Q_load: torch.Tensor,
        u_flow: torch.Tensor,
        n_samples: int = 50,
        confidence_level: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty quantification using MC Dropout.

        Performs multiple forward passes with dropout enabled to estimate
        prediction uncertainty. Returns mean predictions and confidence intervals.

        Args:
            T_current: Current temperature (°C), shape [batch]
            Q_load: IT heat load (W), shape [batch]
            u_flow: Air flow velocity (m/s), shape [batch]
            n_samples: Number of MC dropout samples (default: 50)
            confidence_level: Confidence level for interval (default: 0.95)

        Returns:
            Dictionary with keys:
                - 'T_t1_mean': Mean prediction at t+1, shape [batch, 1]
                - 'T_t1_std': Standard deviation at t+1, shape [batch, 1]
                - 'T_t1_lower': Lower confidence bound, shape [batch, 1]
                - 'T_t1_upper': Upper confidence bound, shape [batch, 1]
                - (same for t5, t10)
                - 'dT_dt_mean': Mean heating rate, shape [batch, 1]
                - 'dT_dt_std': Std of heating rate, shape [batch, 1]
        """
        if not self.use_mc_dropout:
            # If MC Dropout not enabled, return deterministic predictions
            with torch.no_grad():
                outputs = self.forward(T_current, Q_load, u_flow)
            return {
                'T_t1_mean': outputs['T_t1'],
                'T_t1_std': torch.zeros_like(outputs['T_t1']),
                'T_t1_lower': outputs['T_t1'],
                'T_t1_upper': outputs['T_t1'],
                'T_t5_mean': outputs['T_t5'],
                'T_t5_std': torch.zeros_like(outputs['T_t5']),
                'T_t5_lower': outputs['T_t5'],
                'T_t5_upper': outputs['T_t5'],
                'T_t10_mean': outputs['T_t10'],
                'T_t10_std': torch.zeros_like(outputs['T_t10']),
                'T_t10_lower': outputs['T_t10'],
                'T_t10_upper': outputs['T_t10'],
                'dT_dt_mean': outputs['dT_dt'],
                'dT_dt_std': torch.zeros_like(outputs['dT_dt']),
            }

        # Collect predictions from multiple forward passes
        predictions_t1 = []
        predictions_t5 = []
        predictions_t10 = []
        predictions_dT_dt = []

        # Model stays in eval mode, but MCDropout stays active
        self.eval()

        # NOTE: Don't use torch.no_grad() here! MC Dropout needs to regenerate
        # random dropout masks for each forward pass, which requires gradients enabled.
        # Memory usage is acceptable since we're not backpropagating.
        for _ in range(n_samples):
            outputs = self.forward(T_current, Q_load, u_flow)
            predictions_t1.append(outputs['T_t1'].detach())
            predictions_t5.append(outputs['T_t5'].detach())
            predictions_t10.append(outputs['T_t10'].detach())
            predictions_dT_dt.append(outputs['dT_dt'].detach())

        # Stack predictions: [n_samples, batch, 1]
        predictions_t1 = torch.stack(predictions_t1, dim=0)
        predictions_t5 = torch.stack(predictions_t5, dim=0)
        predictions_t10 = torch.stack(predictions_t10, dim=0)
        predictions_dT_dt = torch.stack(predictions_dT_dt, dim=0)

        # Compute statistics
        # Z-score for confidence interval (e.g., 1.96 for 95%)
        z_score = torch.erfinv(torch.tensor(confidence_level)) * np.sqrt(2)

        # t+1 predictions
        T_t1_mean = predictions_t1.mean(dim=0)
        T_t1_std = predictions_t1.std(dim=0)
        T_t1_lower = T_t1_mean - z_score * T_t1_std
        T_t1_upper = T_t1_mean + z_score * T_t1_std

        # t+5 predictions
        T_t5_mean = predictions_t5.mean(dim=0)
        T_t5_std = predictions_t5.std(dim=0)
        T_t5_lower = T_t5_mean - z_score * T_t5_std
        T_t5_upper = T_t5_mean + z_score * T_t5_std

        # t+10 predictions
        T_t10_mean = predictions_t10.mean(dim=0)
        T_t10_std = predictions_t10.std(dim=0)
        T_t10_lower = T_t10_mean - z_score * T_t10_std
        T_t10_upper = T_t10_mean + z_score * T_t10_std

        # dT/dt predictions
        dT_dt_mean = predictions_dT_dt.mean(dim=0)
        dT_dt_std = predictions_dT_dt.std(dim=0)

        return {
            'T_t1_mean': T_t1_mean,
            'T_t1_std': T_t1_std,
            'T_t1_lower': T_t1_lower,
            'T_t1_upper': T_t1_upper,
            'T_t5_mean': T_t5_mean,
            'T_t5_std': T_t5_std,
            'T_t5_lower': T_t5_lower,
            'T_t5_upper': T_t5_upper,
            'T_t10_mean': T_t10_mean,
            'T_t10_std': T_t10_std,
            'T_t10_lower': T_t10_lower,
            'T_t10_upper': T_t10_upper,
            'dT_dt_mean': dT_dt_mean,
            'dT_dt_std': dT_dt_std,
        }


# ============================================================================
# PHYSICS-INFORMED LOSS FUNCTION
# ============================================================================

class RecurrentPINNLoss(nn.Module):
    """
    Physics-informed loss function for Recurrent PINN with Uncertainty Weighting.

    Uses learned uncertainty weighting (Kendall et al., 2018) to automatically
    balance multiple loss components:
        L_total = Σ[(1 / (2·σ_i²)) · L_i + (1/2)·log(σ_i²)]

    This allows the model to learn optimal loss weighting dynamically during training,
    rather than using fixed hyperparameters.

    Where:
        - L_data: MSE between predicted and actual temperatures
        - L_physics: Violation of dT/dt = Q/(m·c_p)
        - L_consistency: Multi-horizon prediction consistency
        - L_monotonicity: Enforce temperature increase during failure
        - σ_i²: Learned variance (uncertainty) for each task
    """

    def __init__(
        self,
        use_uncertainty_weighting: bool = True,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.1,
        lambda_consistency: float = 0.05,
        lambda_monotonicity: float = 0.01,
        thermal_capacity: float = THERMAL_CAPACITY,
        use_stratification: bool = True,
        stratification_coefficient: float = 0.01
    ):
        """
        Initialize physics-informed loss with Height-Dependent Stratification (Step 3.3).

        Args:
            use_uncertainty_weighting: Use learned uncertainty weighting (automatic balancing)
            lambda_data: Weight for data fitting loss (if not using uncertainty weighting)
            lambda_physics: Weight for physics constraint loss (if not using uncertainty weighting)
            lambda_consistency: Weight for multi-horizon consistency (if not using uncertainty weighting)
            lambda_monotonicity: Weight for monotonicity constraint (if not using uncertainty weighting)
            thermal_capacity: Thermal capacity m·c_p (J/K)
            use_stratification: Enable height-dependent stratification penalty (Step 3.3)
            stratification_coefficient: Weight for stratification loss (e.g., 0.01)
        """
        super(RecurrentPINNLoss, self).__init__()

        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.thermal_capacity = thermal_capacity
        self.use_stratification = use_stratification
        self.stratification_coefficient = stratification_coefficient

        if use_uncertainty_weighting:
            # Learnable log-variance parameters for uncertainty weighting
            # Initialize with values that correspond to the fixed lambda weights
            # log_var = log(1/lambda) => lower lambda means higher uncertainty
            self.log_var_data = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0
            self.log_var_physics = nn.Parameter(torch.tensor(np.log(10.0)))  # exp(log(10)) = 10.0
            self.log_var_consistency = nn.Parameter(torch.tensor(np.log(20.0)))  # exp(log(20)) = 20.0
            self.log_var_monotonicity = nn.Parameter(torch.tensor(np.log(100.0)))  # exp(log(100)) = 100.0
        else:
            # Fixed lambda weights (traditional approach)
            self.lambda_data = lambda_data
            self.lambda_physics = lambda_physics
            self.lambda_consistency = lambda_consistency
            self.lambda_monotonicity = lambda_monotonicity

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        Q_load: torch.Tensor,
        u_flow: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss.

        Args:
            predictions: Model predictions with keys ['T_t1', 'T_t5', 'T_t10', 'dT_dt']
            targets: Ground truth with keys ['T_t1', 'T_t5', 'T_t10', 'T_current']
            Q_load: IT heat load (W), shape [batch]
            u_flow: Air flow velocity (m/s), shape [batch]

        Returns:
            Dictionary with total loss and components
        """
        # 1. Data Fitting Loss (MSE)
        loss_data_t1 = F.mse_loss(predictions['T_t1'], targets['T_t1'])
        loss_data_t5 = F.mse_loss(predictions['T_t5'], targets['T_t5'])
        loss_data_t10 = F.mse_loss(predictions['T_t10'], targets['T_t10'])
        loss_data = (loss_data_t1 + loss_data_t5 + loss_data_t10) / 3.0

        # 2. Physics Constraint: dT/dt = Q / (m·c_p)
        # Use learned thermal mass if available (Step 3.2), otherwise use default
        if 'thermal_mass' in predictions:
            # Learned thermal mass from model
            thermal_mass = predictions['thermal_mass']
            thermal_capacity = thermal_mass * C_P_AIR  # C = m * c_p
        else:
            # Default thermal capacity
            thermal_capacity = self.thermal_capacity

        # Compute theoretical heating rate
        dT_dt_theory = Q_load / thermal_capacity  # [batch]

        # During cooling failure, u_flow should be low (cooling reduced/lost)
        # When u_flow is normal, theoretical rate should account for cooling
        # Simplified: if u_flow > 0.5 m/s, assume some cooling
        cooling_active = (u_flow > 0.5).float()

        # Adjust theoretical rate based on cooling status
        # If cooling active, theoretical heating rate is lower
        dT_dt_theory_adjusted = dT_dt_theory * (1.0 - 0.8 * cooling_active)

        # Compare with predicted heating rate
        loss_physics = F.mse_loss(
            predictions['dT_dt'].squeeze(),
            dT_dt_theory_adjusted
        )

        # 3. Multi-Horizon Consistency
        # Predictions should be monotonically ordered: T_t1 <= T_t5 <= T_t10
        # (during heating scenarios)
        T_current = targets['T_current']

        # Compute expected intermediate temperatures using physics
        dt1, dt5, dt10 = 1.0, 5.0, 10.0
        T_t1_expected = T_current + predictions['dT_dt'].squeeze() * dt1
        T_t5_expected = T_current + predictions['dT_dt'].squeeze() * dt5
        T_t10_expected = T_current + predictions['dT_dt'].squeeze() * dt10

        loss_consistency = (
            F.mse_loss(predictions['T_t1'], T_t1_expected.unsqueeze(1)) +
            F.mse_loss(predictions['T_t5'], T_t5_expected.unsqueeze(1)) +
            F.mse_loss(predictions['T_t10'], T_t10_expected.unsqueeze(1))
        ) / 3.0

        # 4. Monotonicity Constraint (during failure)
        # Ensure T_t1 <= T_t5 <= T_t10 when dT/dt > 0
        heating = (predictions['dT_dt'] > 0.0).float()

        violation_1_5 = F.relu(predictions['T_t1'] - predictions['T_t5'])
        violation_5_10 = F.relu(predictions['T_t5'] - predictions['T_t10'])

        loss_monotonicity = (
            (violation_1_5 * heating).mean() +
            (violation_5_10 * heating).mean()
        )

        # ====================================================================
        # STEP 3.3: HEIGHT-DEPENDENT STRATIFICATION
        # ====================================================================
        # Penalize predictions that don't account for vertical temperature gradient
        # Physics: Hot air rises, so ∂T/∂z > 0 (temperature increases with height)
        #
        # Since we don't have explicit z-coordinates, we use a proxy:
        # - Higher temperatures indicate upper positions (heat rises)
        # - Physics loss should account for buoyancy-driven stratification
        #
        # Stratification penalty:  penalize when dT/dt doesn't follow ∂T/∂z
        #
        # We approximate this by enforcing that higher predicted temperatures
        # (which should be at upper heights) have higher heating rates, consistent
        # with vertical stratification

        if self.use_stratification:
            # Compute stratification-aware penalty
            # Assumption: If T is higher, we expect stronger heating rate (more heat accumulation at top)
            # This encourages the model to learn that hot spots develop at upper regions
            T_pred_avg = (predictions['T_t1'] + predictions['T_t5'] + predictions['T_t10']) / 3.0

            # Normalize for numerical stability
            T_normalized = (T_pred_avg - T_current.unsqueeze(1)) / (T_current.unsqueeze(1) + 1e-6)

            # Stratification loss: Penalize inconsistent vertical gradients
            # Higher temperatures should correlate with positive dT/dt (heat rising)
            loss_stratification = F.mse_loss(
                T_normalized.squeeze() * predictions['dT_dt'].squeeze(),
                torch.abs(predictions['dT_dt'].squeeze())  # Expected positive correlation
            ) * self.stratification_coefficient
        else:
            loss_stratification = torch.tensor(0.0, device=T_current.device)

        # Total Loss (with uncertainty weighting or fixed weights)
        if self.use_uncertainty_weighting:
            # Uncertainty weighting: L_weighted = (1 / (2·σ²)) · L + (1/2)·log(σ²)
            # This automatically balances losses based on learned uncertainties
            precision_data = torch.exp(-self.log_var_data)
            precision_physics = torch.exp(-self.log_var_physics)
            precision_consistency = torch.exp(-self.log_var_consistency)
            precision_monotonicity = torch.exp(-self.log_var_monotonicity)

            loss_total = (
                0.5 * precision_data * loss_data + 0.5 * self.log_var_data +
                0.5 * precision_physics * loss_physics + 0.5 * self.log_var_physics +
                0.5 * precision_consistency * loss_consistency + 0.5 * self.log_var_consistency +
                0.5 * precision_monotonicity * loss_monotonicity + 0.5 * self.log_var_monotonicity +
                loss_stratification  # Add stratification penalty (Step 3.3)
            )

            # Store learned weights for logging (inverse of variance)
            learned_weight_data = precision_data.item()
            learned_weight_physics = precision_physics.item()
            learned_weight_consistency = precision_consistency.item()
            learned_weight_monotonicity = precision_monotonicity.item()
        else:
            # Traditional fixed weighting
            loss_total = (
                self.lambda_data * loss_data +
                self.lambda_physics * loss_physics +
                self.lambda_consistency * loss_consistency +
                self.lambda_monotonicity * loss_monotonicity +
                loss_stratification  # Add stratification penalty (Step 3.3)
            )

            learned_weight_data = self.lambda_data
            learned_weight_physics = self.lambda_physics
            learned_weight_consistency = self.lambda_consistency
            learned_weight_monotonicity = self.lambda_monotonicity

        return {
            'loss': loss_total,
            'loss_data': loss_data,
            'loss_physics': loss_physics,
            'loss_consistency': loss_consistency,
            'loss_monotonicity': loss_monotonicity,
            'loss_stratification': loss_stratification,  # Add for monitoring (Step 3.3)
            'loss_data_t1': loss_data_t1,
            'loss_data_t5': loss_data_t5,
            'loss_data_t10': loss_data_t10,
            # Learned weights (for logging)
            'weight_data': learned_weight_data,
            'weight_physics': learned_weight_physics,
            'weight_consistency': learned_weight_consistency,
            'weight_monotonicity': learned_weight_monotonicity,
        }


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    for horizon in ['t1', 't5', 't10']:
        pred_key = f'T_{horizon}'
        target_key = f'T_{horizon}'

        pred = predictions[pred_key].detach()
        target = targets[target_key].detach()

        # Mean Absolute Error (MAE)
        mae = torch.mean(torch.abs(pred - target)).item()
        metrics[f'mae_{horizon}'] = mae

        # Root Mean Squared Error (RMSE)
        rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
        metrics[f'rmse_{horizon}'] = rmse

        # Mean Absolute Percentage Error (MAPE)
        mape = torch.mean(torch.abs((pred - target) / (target + 1e-8)) * 100).item()
        metrics[f'mape_{horizon}'] = mape

        # R² Score
        ss_res = torch.sum((pred - target) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        metrics[f'r2_{horizon}'] = r2.item()

    # Average metrics
    metrics['mae_avg'] = np.mean([metrics[f'mae_{h}'] for h in ['t1', 't5', 't10']])
    metrics['rmse_avg'] = np.mean([metrics[f'rmse_{h}'] for h in ['t1', 't5', 't10']])
    metrics['mape_avg'] = np.mean([metrics[f'mape_{h}'] for h in ['t1', 't5', 't10']])

    return metrics


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: str = 'cpu'
) -> int:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


# ============================================================================
# MODEL SUMMARY
# ============================================================================

def print_model_summary(model: RecurrentPINN):
    """Print detailed model summary."""
    print("=" * 70)
    print("RECURRENT PINN MODEL SUMMARY")
    print("=" * 70)
    print(f"\nArchitecture:")
    print(f"  Input dimension: {model.input_dim}")
    print(f"  Hidden dimension: {model.hidden_dim}")
    print(f"  LSTM layers: {model.num_lstm_layers}")
    print(f"  Total parameters: {count_parameters(model):,}")

    print(f"\nInput Features:")
    print(f"  - T_current: Current temperature (°C)")
    print(f"  - Q_load: IT equipment heat load (W)")
    print(f"  - u_flow: Air flow velocity (m/s)")

    print(f"\nOutput Predictions:")
    print(f"  - T_t1: Temperature 1 second ahead")
    print(f"  - T_t5: Temperature 5 seconds ahead")
    print(f"  - T_t10: Temperature 10 seconds ahead")
    print(f"  - dT_dt: Heating rate (°C/s)")

    print(f"\nPhysics Constraints:")
    print(f"  - Thermal capacity: {THERMAL_CAPACITY / 1e6:.2f} MJ/K")
    print(f"  - Thermal mass: {THERMAL_MASS:.1f} kg")
    print(f"  - Heat equation: dT/dt = Q / (m·c_p)")

    print("=" * 70)


if __name__ == "__main__":
    # Test model creation
    print("Testing Recurrent PINN...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = RecurrentPINN(
        input_dim=3,
        hidden_dim=64,
        num_lstm_layers=2,
        num_dense_layers=3,
        dropout=0.1
    ).to(device)

    print_model_summary(model)

    # Test forward pass
    batch_size = 8
    T_current = torch.randn(batch_size, device=device) * 10 + 25  # ~15-35°C
    Q_load = torch.randn(batch_size, device=device) * 20000 + 100000  # ~80-120kW
    u_flow = torch.randn(batch_size, device=device) * 0.5 + 1.5  # ~1-2 m/s

    print(f"\nTest forward pass...")
    print(f"  Batch size: {batch_size}")
    print(f"  Input shapes: T={T_current.shape}, Q={Q_load.shape}, u={u_flow.shape}")

    with torch.no_grad():
        outputs = model(T_current, Q_load, u_flow)

    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    print(f"\nSample predictions (first example):")
    print(f"  Input: T={T_current[0].item():.2f}°C, Q={Q_load[0].item()/1000:.1f}kW, u={u_flow[0].item():.2f}m/s")
    print(f"  T_t1:  {outputs['T_t1'][0].item():.2f}°C")
    print(f"  T_t5:  {outputs['T_t5'][0].item():.2f}°C")
    print(f"  T_t10: {outputs['T_t10'][0].item():.2f}°C")
    print(f"  dT/dt: {outputs['dT_dt'][0].item():.4f}°C/s")

    # Test Time-to-Failure estimation
    print(f"\nTest Time-to-Failure estimation...")
    ttf = model.estimate_time_to_failure(
        T_current[:4],
        Q_load[:4],
        u_flow[:4],
        threshold=T_CRITICAL,
        max_time=60
    )
    print(f"  Time-to-Critical (85°C): {ttf.tolist()}")

    print("\n✓ Recurrent PINN test completed successfully!")
    print("=" * 70)
