from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging
import traceback
from datetime import datetime

from schemas import (
    GANTrainingRequest, GANTrainingResponse,
    DataGenerationRequest, GenerationResponse
)
from gan_trainer import GANTrainer
from generate import DiabetesDataGenerator
from config import DEFAULT_TIME_SERIES_PATH, DEFAULT_TABULAR_PATH

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Global instances
gan_trainer_instance = GANTrainer()
generator = DiabetesDataGenerator()

# ==================== GAN TRAINING (ONLY ENDPOINT) ====================
@router.post("/train/gan", response_model=GANTrainingResponse)
async def train_gan_models(request: GANTrainingRequest):
    """
    Train GAN models for synthetic data generation using GitHub datasets.
    This is the ONLY training endpoint available.
    """
    try:
        logger.info("=" * 80)
        logger.info("Starting GAN model training...")
        logger.info(f"Epochs: {request.epochs} | Batch size: {request.batch_size}")
        logger.info(f"Using GitHub dataset URLs (no local fallback)")
        logger.info("=" * 80)

        # ALWAYS use GitHub URLs (no local override)
        ts_path = DEFAULT_TIME_SERIES_PATH
        tab_path = DEFAULT_TABULAR_PATH

        # Train GAN models
        history = gan_trainer_instance.train_gan(
            time_series_path=ts_path,
            tabular_path=tab_path,
            epochs=request.epochs
        )

        return GANTrainingResponse(
            status="success",
            message="GAN model training completed successfully",
            epochs_completed=request.epochs,
            training_metrics=history,
            model_timestamp=datetime.now().isoformat()
        )

    except RuntimeError as e:
        # Dataset loading error (GitHub fetch failed)
        error_msg = str(e)
        if "Failed to load dataset from GitHub" in error_msg:
            logger.error(f"Dataset fetch error: {error_msg}")
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "error",
                    "message": "Failed to load dataset from GitHub",
                    "url": ts_path if "time series" in error_msg.lower() else tab_path
                }
            )
        raise HTTPException(status_code=500, detail=f"Training failed: {error_msg}")
    
    except Exception as e:
        logger.error(f"GAN training failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"GAN training failed: {str(e)}")

# ==================== REJECT ALL OTHER TRAINING ROUTES ====================
@router.post("/train")
@router.post("/train/model")
@router.post("/train/supervised")
async def reject_other_training():
    """Reject all non-GAN training endpoints."""
    raise HTTPException(
        status_code=405,
        detail={
            "error": "Only gan-train is supported in this deployment. Use /api/v1/train/gan."
        }
    )

# ==================== DATA GENERATION ====================
@router.post("/generate", response_model=GenerationResponse)
async def generate_synthetic_data(request: DataGenerationRequest):
    """
    Generate synthetic diabetes data using trained GAN models.
    REQUIRES GAN models to be trained first. NO STATISTICAL FALLBACK.
    """
    try:
        logger.info(f"Generating {request.num_samples} synthetic samples...")

        # Check if GAN models are loaded
        if not generator.models_loaded:
            logger.error("GAN models not available for generation")
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "error",
                    "message": "GAN models not available. Please train using /api/v1/train/gan first."
                }
            )

        # Generate synthetic data using GAN
        result = generator.generate_synthetic_data(
            num_samples=request.num_samples,
            diabetes_ratio=request.diabetes_ratio,
            hypertension_ratio=request.hypertension_ratio
        )

        return GenerationResponse(
            status="success",
            message=f"Successfully generated {request.num_samples} synthetic samples using GAN",
            num_generated=request.num_samples,
            timeseries_file=result['timeseries_file'],
            tabular_file=result['tabular_file'],
            preview=result['preview']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data generation failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")

# ==================== MODEL STATUS ====================
@router.get("/models/status")
async def get_model_status():
    """Get current GAN model training status."""
    try:
        gan_models_loaded = generator.models_loaded

        return {
            "status": "success",
            "gan_models_loaded": gan_models_loaded,
            "generation_method": "GAN" if gan_models_loaded else "Not Available",
            "training_endpoint": "/api/v1/train/gan",
            "last_updated": datetime.now().isoformat(),
            "dataset_paths": {
                "time_series": DEFAULT_TIME_SERIES_PATH,
                "tabular": DEFAULT_TABULAR_PATH
            }
        }
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# ==================== HEALTH CHECK ====================
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
