from pydantic import BaseModel, Field, validator, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum

# ==================== ENUMS ====================
class DiabetesStatusEnum(str, Enum):
    """Diabetes status classification."""
    non_diabetic = "Non-Diabetic"
    diabetic = "Diabetic"

class BPStatusEnum(str, Enum):
    """Blood pressure status classification."""
    normal = "Normal"
    hypertensive = "Hypertensive"

# ==================== GAN TRAINING SCHEMAS ====================
class GANTrainingRequest(BaseModel):
    """
    Request for GAN model training (Synthetic Data Generation).
    This trains GANs that generate realistic synthetic patient data.
    """
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "epochs": 100,
                "batch_size": 32
            }
        }
    )
    
    epochs: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Training epochs (1-500). Recommended: 100-200",
        example=100
    )
    
    batch_size: int = Field(
        default=32,
        ge=8,
        le=128,
        description="Batch size (8-128). Recommended: 32",
        example=32
    )
    
    @validator('epochs')
    def validate_epochs(cls, v):
        if not 1 <= v <= 500:
            raise ValueError('Epochs must be between 1 and 500')
        return v
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        if not 8 <= v <= 128:
            raise ValueError('Batch size must be between 8 and 128')
        return v

# ==================== DATA GENERATION SCHEMAS ====================
class DataGenerationRequest(BaseModel):
    """Request for synthetic diabetes data generation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "num_samples": 500,
                "diabetes_ratio": 0.5,
                "hypertension_ratio": 0.7
            }
        }
    )
    
    num_samples: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of synthetic patients to generate (1-10,000)",
        example=500
    )
    
    diabetes_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ratio of diabetic patients (0.0-1.0)",
        example=0.5
    )
    
    hypertension_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Ratio of hypertensive patients (0.0-1.0)",
        example=0.7
    )
    
    @validator('num_samples')
    def validate_num_samples(cls, v):
        if not 1 <= v <= 10000:
            raise ValueError('Number of samples must be between 1 and 10,000')
        return v

# ==================== RESPONSE SCHEMAS ====================
class GANTrainingResponse(BaseModel):
    """Response after GAN training completion."""
    status: str = Field(..., description="Training status", example="success")
    message: str = Field(..., description="Training completion message")
    epochs_completed: int = Field(..., description="Number of epochs completed")
    training_metrics: Optional[Dict[str, List[float]]] = Field(None, description="GAN training history")
    model_timestamp: Optional[str] = Field(None, description="Model training timestamp")

class GenerationResponse(BaseModel):
    """Response after synthetic data generation."""
    status: str = Field(..., description="Generation status", example="success")
    message: str = Field(..., description="Generation completion message")
    num_generated: int = Field(..., description="Number of samples generated")
    timeseries_file: str = Field(..., description="Path to generated time series CSV file")
    tabular_file: str = Field(..., description="Path to generated tabular CSV file")
    preview: Dict[str, Any] = Field(..., description="Preview of generated data")

class ErrorResponse(BaseModel):
    """Error response schema for failed requests."""
    status: str = Field(default="error", description="Error status")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
