# Ka-MLOps API Schemas (Fixed for Pydantic v2)
from pydantic import BaseModel, Field
from typing import Optional

class KaLoanRequest(BaseModel):
    '''Ka Lending Club loan request schema'''
    loan_amnt: float = Field(..., description="Loan amount", ge=1000, le=40000)
    int_rate: float = Field(..., description="Interest rate", ge=5.0, le=30.0)
    annual_inc: float = Field(..., description="Annual income", ge=4000)
    dti: float = Field(..., description="Debt-to-income ratio", ge=0, le=40)
    fico_range_low: int = Field(..., description="FICO low", ge=300, le=850)
    fico_range_high: int = Field(..., description="FICO high", ge=300, le=850)
    installment: float = Field(..., description="Monthly installment", ge=0)
    delinq_2yrs: int = Field(..., description="Delinquencies in past 2 years", ge=0)
    inq_last_6mths: int = Field(..., description="Inquiries in last 6 months", ge=0)
    open_acc: int = Field(..., description="Number of open accounts", ge=0)
    pub_rec: int = Field(..., description="Number of public records", ge=0)
    revol_bal: float = Field(..., description="Revolving balance", ge=0)
    revol_util: float = Field(..., description="Revolving utilization", ge=0, le=100)
    total_acc: int = Field(..., description="Total accounts", ge=0)
    mort_acc: int = Field(..., description="Mortgage accounts", ge=0)
    pub_rec_bankruptcies: int = Field(..., description="Public bankruptcies", ge=0)
    
    # Categorical features (fixed pattern syntax)
    term: str = Field(..., description="Loan term", pattern=r"^( 36 months| 60 months)$")
    grade: str = Field(..., description="Loan grade", pattern=r"^[A-G]$")
    emp_length: str = Field(..., description="Employment length")
    home_ownership: str = Field(..., description="Home ownership status")
    verification_status: str = Field(..., description="Income verification status")
    purpose: str = Field(..., description="Loan purpose")
    addr_state: str = Field(..., description="State abbreviation", min_length=2, max_length=2)

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 15000.0,
                "int_rate": 12.5,
                "annual_inc": 65000.0,
                "dti": 18.5,
                "fico_range_low": 720,
                "fico_range_high": 724,
                "installment": 450.0,
                "delinq_2yrs": 0,
                "inq_last_6mths": 1,
                "open_acc": 8,
                "pub_rec": 0,
                "revol_bal": 5000.0,
                "revol_util": 25.0,
                "total_acc": 15,
                "mort_acc": 1,
                "pub_rec_bankruptcies": 0,
                "term": " 36 months",
                "grade": "B",
                "emp_length": "5 years",
                "home_ownership": "MORTGAGE",
                "verification_status": "Verified",
                "purpose": "debt_consolidation",
                "addr_state": "CA"
            }
        }

class KaPredictionResponse(BaseModel):
    '''Ka prediction response schema'''
    prediction: str = Field(..., description="Prediction: 'approved' or 'rejected'")
    default_probability: float = Field(..., description="Default probability", ge=0, le=1)
    confidence: str = Field(..., description="Confidence: 'high', 'medium', 'low'")
    risk_factors: list = Field(..., description="Top risk factors")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "approved",
                "default_probability": 0.15,
                "confidence": "high",
                "risk_factors": ["low_fico_score", "high_dti"]
            }
        }

class KaHealthResponse(BaseModel):
    '''Ka health check response'''
    status: str = "healthy"
    timestamp: str
    version: str = "1.0.0"
    model_loaded: bool
    ka_system: str = "operational"
