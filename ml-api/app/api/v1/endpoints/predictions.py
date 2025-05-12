from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from app.utils.model_loader import get_model_manager, ModelManager

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for request/response
class StudentFeatures(BaseModel):
    """Input features for prediction"""
    age: int = Field(..., ge=16, le=25, description="Student age")
    gender: str = Field(..., regex="^(Male|Female)$", description="Student gender")
    academic_level: str = Field(..., regex="^(High School|Undergraduate|Graduate)$", description="Academic level")
    country: str = Field(..., description="Country of residence")
    avg_daily_usage_hours: float = Field(..., ge=0, le=24, description="Average daily social media usage in hours")
    most_used_platform: str = Field(..., description="Most used social media platform")
    sleep_hours_per_night: float = Field(..., ge=0, le=24, description="Average sleep hours per night")
    relationship_status: str = Field(..., regex="^(Single|In Relationship|Complicated)$", description="Relationship status")
    conflicts_over_social_media: int = Field(..., ge=0, description="Number of conflicts due to social media")


class PredictionResponse(BaseModel):
    """Standard prediction response"""
    prediction: Any
    confidence: Optional[float] = None
    model_version: str
    timestamp: str
    request_id: str


class AcademicPerformanceResponse(PredictionResponse):
    """Response for academic performance prediction"""
    prediction: str = Field(..., description="Yes or No")
    probability: Dict[str, float] = Field(..., description="Probability for each class")


class MentalHealthResponse(PredictionResponse):
    """Response for mental health score prediction"""
    prediction: float = Field(..., ge=1, le=10, description="Mental health score (1-10)")
    prediction_interval: Dict[str, float] = Field(..., description="Confidence interval")


class SleepPredictionResponse(PredictionResponse):
    """Response for sleep hours prediction"""
    prediction: float = Field(..., ge=0, le=24, description="Predicted sleep hours")
    prediction_interval: Dict[str, float] = Field(..., description="Confidence interval")


class AddictionRiskResponse(PredictionResponse):
    """Response for addiction risk prediction"""
    prediction: int = Field(..., ge=1, le=10, description="Addiction score (1-10)")
    risk_level: str = Field(..., description="Low, Medium, or High risk")
    probability: Dict[str, float] = Field(..., description="Probability for each risk level")


@router.post("/academic-performance", response_model=AcademicPerformanceResponse)
async def predict_academic_performance(
    features: StudentFeatures,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Predict if social media affects academic performance"""
    try:
        # Get model
        model = model_manager.get_model("academic_performance_classifier")
        
        # Prepare features for prediction
        feature_dict = features.dict()
        
        # Make prediction
        prediction, probabilities = model.predict(feature_dict)
        
        return AcademicPerformanceResponse(
            prediction=prediction,
            probability=probabilities,
            confidence=max(probabilities.values()),
            model_version=model.version,
            timestamp=datetime.utcnow().isoformat(),
            request_id=f"acad_{datetime.utcnow().timestamp()}"
        )
    except Exception as e:
        logger.error(f"Academic performance prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mental-health", response_model=MentalHealthResponse)
async def predict_mental_health(
    features: StudentFeatures,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Predict mental health score"""
    try:
        # Get model
        model = model_manager.get_model("mental_health_predictor")
        
        # Prepare features for prediction
        feature_dict = features.dict()
        
        # Make prediction
        prediction, interval = model.predict(feature_dict)
        
        return MentalHealthResponse(
            prediction=prediction,
            prediction_interval=interval,
            confidence=None,  # Regression models don't have confidence in the same way
            model_version=model.version,
            timestamp=datetime.utcnow().isoformat(),
            request_id=f"mental_{datetime.utcnow().timestamp()}"
        )
    except Exception as e:
        logger.error(f"Mental health prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sleep-pattern", response_model=SleepPredictionResponse)
async def predict_sleep_hours(
    features: StudentFeatures,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Predict sleep hours based on social media usage"""
    try:
        # Get model
        model = model_manager.get_model("sleep_predictor")
        
        # Prepare features for prediction
        feature_dict = features.dict()
        
        # Make prediction
        prediction, interval = model.predict(feature_dict)
        
        return SleepPredictionResponse(
            prediction=prediction,
            prediction_interval=interval,
            confidence=None,
            model_version=model.version,
            timestamp=datetime.utcnow().isoformat(),
            request_id=f"sleep_{datetime.utcnow().timestamp()}"
        )
    except Exception as e:
        logger.error(f"Sleep prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/addiction-risk", response_model=AddictionRiskResponse)
async def predict_addiction_risk(
    features: StudentFeatures,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Calculate addiction risk level"""
    try:
        # Get model
        model = model_manager.get_model("addiction_classifier")
        
        # Prepare features for prediction
        feature_dict = features.dict()
        
        # Make prediction
        prediction, probabilities = model.predict(feature_dict)
        
        # Convert to risk level
        if prediction <= 3:
            risk_level = "Low"
        elif prediction <= 7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return AddictionRiskResponse(
            prediction=prediction,
            risk_level=risk_level,
            probability=probabilities,
            confidence=max(probabilities.values()),
            model_version=model.version,
            timestamp=datetime.utcnow().isoformat(),
            request_id=f"addiction_{datetime.utcnow().timestamp()}"
        )
    except Exception as e:
        logger.error(f"Addiction risk prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Additional endpoints for model management
@router.get("/models/health")
async def check_models_health(model_manager: ModelManager = Depends(get_model_manager)):
    """Check health of all loaded models"""
    return {
        "status": "healthy",
        "models": model_manager.get_models_info(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/models/performance")
async def get_model_performance(model_manager: ModelManager = Depends(get_model_manager)):
    """Get performance metrics for all models"""
    return model_manager.get_performance_metrics()