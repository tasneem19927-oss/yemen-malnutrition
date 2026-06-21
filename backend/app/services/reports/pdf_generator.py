"""
PDF Report Generator for clinical reports.
Supports Arabic and English bilingual reports.
"""

import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_RIGHT, TA_CENTER, TA_LEFT


class PDFReportGenerator:
    """Generate bilingual PDF reports for malnutrition predictions."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """Setup custom styles for Arabic and English."""
        # Arabic style (RTL)
        self.styles.add(ParagraphStyle(
            name='Arabic',
            fontName='Helvetica',
            fontSize=11,
            leading=14,
            alignment=TA_RIGHT,
            rightIndent=10,
            spaceAfter=6,
        ))

        self.styles.add(ParagraphStyle(
            name='ArabicTitle',
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=colors.HexColor('#0066CC'),
        ))

        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=16,
            textColor=colors.HexColor('#0066CC'),
            spaceAfter=8,
            spaceBefore=12,
        ))

        self.styles.add(ParagraphStyle(
            name='RiskSevere',
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=colors.red,
        ))

        self.styles.add(ParagraphStyle(
            name='RiskModerate',
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=colors.orange,
        ))

        self.styles.add(ParagraphStyle(
            name='RiskMild',
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=colors.HexColor('#FFCC00'),
        ))

        self.styles.add(ParagraphStyle(
            name='RiskNormal',
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=colors.green,
        ))

    async def generate(
        self,
        prediction: Any,
        language: str = "en",
        include_evidence: bool = True,
        include_recommendations: bool = True,
        include_doctor_notes: bool = True,
    ) -> str:
        """
        Generate PDF report.

        Returns:
            Path to generated PDF file
        """
        # Create temp file
        temp_dir = tempfile.gettempdir()
        filename = f"malnutrition_report_{prediction.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(temp_dir, filename)

        # Build document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm,
        )

        story = []

        # Header
        story.extend(self._build_header(prediction, language))

        # Patient Information
        story.extend(self._build_patient_section(prediction, language))

        # Anthropometric Measures
        story.extend(self._build_measurements_section(prediction, language))

        # Prediction Results
        story.extend(self._build_prediction_section(prediction, language))

        # Evidence
        if include_evidence and prediction.rag_evidence:
            story.extend(self._build_evidence_section(prediction, language))

        # Recommendations
        if include_recommendations:
            story.extend(self._build_recommendations_section(prediction, language))

        # Doctor Notes
        if include_doctor_notes and prediction.doctor_notes:
            story.extend(self._build_doctor_notes_section(prediction, language))

        # Footer
        story.extend(self._build_footer(prediction, language))

        # Build PDF
        doc.build(story)

        return filepath

    def _build_header(self, prediction: Any, language: str) -> list:
        """Build report header."""
        elements = []

        if language == "ar":
            title = "تقرير التنبؤ بسوء التغذية"
            subtitle = "نظام الدعم السريري للقرارات - اليمن"
        else:
            title = "Malnutrition Prediction Report"
            subtitle = "Clinical Decision Support System - Yemen"

        elements.append(Paragraph(title, self.styles['ArabicTitle'] if language == "ar" else self.styles['Title']))
        elements.append(Paragraph(subtitle, self.styles['Heading2']))
        elements.append(Spacer(1, 0.5*cm))

        # Report metadata
        meta_data = [
            ["Report ID:", f"RPT-{prediction.id:06d}"],
            ["Date:", datetime.utcnow().strftime("%Y-%m-%d %H:%M")],
            ["Model Version:", prediction.model_version or "1.0.0"],
        ]

        meta_table = Table(meta_data, colWidths=[4*cm, 10*cm])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F0F0')),
        ]))

        elements.append(meta_table)
        elements.append(Spacer(1, 0.5*cm))

        return elements

    def _build_patient_section(self, prediction: Any, language: str) -> list:
        """Build patient information section."""
        elements = []
        patient = prediction.patient

        if language == "ar":
            title = "معلومات المريض"
        else:
            title = "Patient Information"

        elements.append(Paragraph(title, self.styles['SectionHeader']))

        patient_data = [
            ["Name:", f"{patient.first_name} {patient.last_name}"],
            ["Registration:", patient.registration_number],
            ["Age:", f"{patient.age_months} months"],
            ["Sex:", patient.sex],
            ["Caregiver:", patient.caregiver_name or "N/A"],
            ["Location:", f"{patient.governorate or ''}, {patient.district or ''}"],
        ]

        table = Table(patient_data, colWidths=[4*cm, 10*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F0F0')),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.5*cm))

        return elements

    def _build_measurements_section(self, prediction: Any, language: str) -> list:
        """Build anthropometric measurements section."""
        elements = []
        measurement = prediction.measurement

        if language == "ar":
            title = "القياسات البشرية"
        else:
            title = "Anthropometric Measures"

        elements.append(Paragraph(title, self.styles['SectionHeader']))

        measures = [
            ["Weight (kg):", f"{measurement.weight_kg:.2f}"],
            ["Height (cm):", f"{measurement.height_cm:.2f}"],
            ["MUAC (mm):", f"{measurement.muac_mm or 'N/A'}"],
            ["HAZ:", f"{measurement.haz or 'N/A'}"],
            ["WHZ:", f"{measurement.whz or 'N/A'}"],
            ["WAZ:", f"{measurement.waz or 'N/A'}"],
            ["Oedema:", "Yes" if measurement.oedema else "No"],
        ]

        table = Table(measures, colWidths=[4*cm, 10*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F0F0')),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.5*cm))

        return elements

    def _build_prediction_section(self, prediction: Any, language: str) -> list:
        """Build prediction results section."""
        elements = []

        if language == "ar":
            title = "نتائج التنبؤ"
        else:
            title = "Prediction Results"

        elements.append(Paragraph(title, self.styles['SectionHeader']))

        # Stunting
        stunting_style = self._get_risk_style(prediction.stunting_severity)
        stunting_text = f"Stunting: {prediction.stunting_severity.upper()} (Risk: {prediction.stunting_risk_percent:.1f}%, Confidence: {prediction.stunting_confidence:.1%})"
        elements.append(Paragraph(stunting_text, stunting_style))

        # Wasting
        wasting_style = self._get_risk_style(prediction.wasting_severity)
        wasting_text = f"Wasting: {prediction.wasting_severity.upper()} (Risk: {prediction.wasting_risk_percent:.1f}%, Confidence: {prediction.wasting_confidence:.1%})"
        elements.append(Paragraph(wasting_text, wasting_style))

        # Underweight
        underweight_style = self._get_risk_style(prediction.underweight_severity)
        underweight_text = f"Underweight: {prediction.underweight_severity.upper()} (Risk: {prediction.underweight_risk_percent:.1f}%, Confidence: {prediction.underweight_confidence:.1%})"
        elements.append(Paragraph(underweight_text, underweight_style))

        elements.append(Spacer(1, 0.5*cm))

        # Overall
        overall_style = self._get_risk_style(prediction.overall_risk)
        overall_text = f"Overall Risk: {prediction.overall_risk.upper()}"
        elements.append(Paragraph(overall_text, overall_style))

        elements.append(Spacer(1, 0.5*cm))

        return elements

    def _get_risk_style(self, severity: str):
        """Get style based on severity."""
        severity_map = {
            "severe": self.styles['RiskSevere'],
            "moderate": self.styles['RiskModerate'],
            "mild": self.styles['RiskMild'],
            "normal": self.styles['RiskNormal'],
        }
        return severity_map.get(severity, self.styles['Normal'])

    def _build_evidence_section(self, prediction: Any, language: str) -> list:
        """Build evidence section."""
        elements = []

        if language == "ar":
            title = "الأدلة السريرية"
        else:
            title = "Clinical Evidence"

        elements.append(Paragraph(title, self.styles['SectionHeader']))

        evidence = prediction.rag_evidence or []
        for i, ev in enumerate(evidence[:3], 1):
            text = f"{i}. {ev.get('title', 'N/A')} - {ev.get('citation', 'N/A')}"
            elements.append(Paragraph(text, self.styles['BodyText']))

        elements.append(Spacer(1, 0.5*cm))

        return elements

    def _build_recommendations_section(self, prediction: Any, language: str) -> list:
        """Build recommendations section."""
        elements = []

        if language == "ar":
            title = "التوصيات العلاجية"
        else:
            title = "Treatment Recommendations"

        elements.append(Paragraph(title, self.styles['SectionHeader']))

        rec = prediction.recommended_intervention or "No specific recommendation."
        elements.append(Paragraph(rec, self.styles['BodyText']))

        if prediction.referral_needed:
            if language == "ar":
                ref_text = f"إحالة مطلوبة: {prediction.referral_urgency or 'routine'}"
            else:
                ref_text = f"Referral needed: {prediction.referral_urgency or 'routine'}"
            elements.append(Paragraph(ref_text, self.styles['RiskSevere']))

        elements.append(Spacer(1, 0.5*cm))

        return elements

    def _build_doctor_notes_section(self, prediction: Any, language: str) -> list:
        """Build doctor notes section."""
        elements = []

        if language == "ar":
            title = "ملاحظات الطبيب"
        else:
            title = "Doctor Notes"

        elements.append(Paragraph(title, self.styles['SectionHeader']))
        elements.append(Paragraph(prediction.doctor_notes or "No notes.", self.styles['BodyText']))
        elements.append(Spacer(1, 0.5*cm))

        return elements

    def _build_footer(self, prediction: Any, language: str) -> list:
        """Build report footer."""
        elements = []

        if language == "ar":
            disclaimer = """
            هذا التقرير تم إنشاؤه بواسطة نظام الذكاء الاصطناعي ولا يغني عن التشخيص السريري.
            يرجى استشارة أخصائي صحي مؤهل.
            """
        else:
            disclaimer = """
            This report was generated by an AI system and does not replace clinical diagnosis.
            Please consult a qualified healthcare professional.
            """

        elements.append(Spacer(1, 1*cm))
        elements.append(Paragraph(disclaimer, self.styles['Italic']))

        return elements
