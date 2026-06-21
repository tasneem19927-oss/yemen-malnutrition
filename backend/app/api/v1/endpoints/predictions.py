"""
Prediction and Clinical Reasoning endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.session import get_db
from app.core.security import require_doctor, require_nurse, get_current_user
from app.models.user import User
from app.models.patient import Patient, Measurement
from app.models.prediction import Prediction, ClinicalNote
from app.schemas.prediction import (
    PredictionRequest, PredictionResult, PredictionInput,
    DoctorReviewRequest, ClinicalNoteCreate, ClinicalNoteResponse,
    ReportRequest
)
from app.services.ml.prediction_engine import PredictionEngine
from app.services.rag.clinical_rag import ClinicalRAG
from app.services.reports.pdf_generator import PDFReportGenerator

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("/predict", response_model=PredictionResult)
async def create_prediction(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_nurse),
):
    """Run malnutrition prediction for a patient."""
    # Get patient and measurement
    patient = db.query(Patient).filter(Patient.id == request.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    measurement = db.query(Measurement).filter(Measurement.id == request.measurement_id).first()
    if not measurement or measurement.patient_id != request.patient_id:
        raise HTTPException(status_code=404, detail="Measurement not found")

    # Prepare input features
    prediction_input = PredictionInput(
        patient_id=request.patient_id,
        measurement_id=request.measurement_id,
        age_months=patient.age_months,
        sex=patient.sex,
        weight_kg=measurement.weight_kg,
        height_cm=measurement.height_cm,
        oedema=measurement.oedema,
        breastfeeding=measurement.breastfeeding,
        vitamin_a=measurement.vitamin_a,
        diarrhea_recent=measurement.diarrhea_recent,
        fever_recent=measurement.fever_recent,
        cough_recent=measurement.cough_recent,
        maternal_education=patient.maternal_education,
        wealth_index=patient.wealth_index,
        residence_type=patient.residence_type,
        haz=measurement.haz,
        whz=measurement.whz,
        waz=measurement.waz,
    )

    # Run prediction engine
    engine = PredictionEngine()
    result = await engine.predict(prediction_input)

    # Clinical reasoning with RAG
    if request.include_rag:
        rag = ClinicalRAG()
        rag_result = await rag.get_recommendations(
            prediction_result=result,
            patient=patient,
            measurement=measurement,
            language=request.language,
        )
        result.rag_evidence = rag_result["evidence"]
        result.recommended_intervention = rag_result["recommendation"]
        result.referral_needed = rag_result["referral_needed"]
        result.referral_urgency = rag_result["referral_urgency"]

    # Save prediction
    prediction = Prediction(
        patient_id=request.patient_id,
        measurement_id=request.measurement_id,
        created_by=current_user.id,
        input_features=result.input_features,
        stunting_probability=result.stunting.probability,
        stunting_risk_percent=result.stunting.risk_percent,
        stunting_severity=result.stunting.severity,
        stunting_confidence=result.stunting.confidence,
        wasting_probability=result.wasting.probability,
        wasting_risk_percent=result.wasting.risk_percent,
        wasting_severity=result.wasting.severity,
        wasting_confidence=result.wasting.confidence,
        underweight_probability=result.underweight.probability,
        underweight_risk_percent=result.underweight.risk_percent,
        underweight_severity=result.underweight.severity,
        underweight_confidence=result.underweight.confidence,
        overall_risk=result.overall_risk,
        overall_recommendation=result.overall_recommendation,
        clinical_query=result.clinical_query,
        rag_evidence=result.rag_evidence,
        recommended_intervention=result.recommended_intervention,
        referral_needed=result.referral_needed,
        referral_urgency=result.referral_urgency,
        model_version=result.model_version,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return PredictionResult.model_validate(prediction)


@router.get("/patient/{patient_id}", response_model=List[PredictionResult])
async def get_patient_predictions(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_nurse),
):
    """Get all predictions for a patient."""
    predictions = db.query(Prediction).filter(Prediction.patient_id == patient_id).order_by(Prediction.created_at.desc()).all()
    return [PredictionResult.model_validate(p) for p in predictions]


@router.post("/{prediction_id}/review")
async def review_prediction(
    prediction_id: int,
    review: DoctorReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor),
):
    """Doctor reviews and approves/rejects prediction."""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    prediction.doctor_notes = review.doctor_notes
    prediction.doctor_approved = review.approved
    prediction.approved_at = datetime.utcnow() if review.approved else None

    if review.override_recommendation:
        prediction.recommended_intervention = review.override_recommendation

    db.commit()
    return {"message": "Review submitted successfully"}


# Clinical Notes
@router.post("/notes", response_model=ClinicalNoteResponse)
async def add_clinical_note(
    note_data: ClinicalNoteCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor),
):
    """Add clinical note with NER extraction."""
    from app.services.nlp.biomobilebert_ner import BioMobileBERTNER

    ner_service = BioMobileBERTNER.get_instance()
    ner_result = ner_service.extract_medical_entities(note_data.content)

    note = ClinicalNote(
        patient_id=note_data.patient_id,
        created_by=current_user.id,
        note_type=note_data.note_type,
        content=note_data.content,
        content_ar=note_data.content_ar,
        extracted_entities=[e.to_dict() for e in ner_result.entities],
    )
    db.add(note)
    db.commit()
    db.refresh(note)

    return ClinicalNoteResponse.model_validate(note)


# Reports
@router.post("/reports/generate")
async def generate_report(
    request: ReportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor),
):
    """Generate PDF report for a prediction."""
    prediction = db.query(Prediction).filter(Prediction.id == request.prediction_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    generator = PDFReportGenerator()
    pdf_path = await generator.generate(
        prediction=prediction,
        language=request.language,
        include_evidence=request.include_evidence,
        include_recommendations=request.include_recommendations,
        include_doctor_notes=request.include_doctor_notes,
    )

    from fastapi.responses import FileResponse
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"malnutrition_report_{prediction.id}.pdf",
    )
