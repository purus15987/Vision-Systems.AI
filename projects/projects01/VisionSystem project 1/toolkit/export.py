"""
Report Generation, CSV Metrics Export, and ZIP Archiving Module.
Generates comprehensive PDF Technical Reports using ReportLab,
exports quantitative data to CSV, and bundles results into ZIP archives.
"""

import os
import csv
import zipfile
import io
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from PIL import Image

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def export_metrics_to_csv(metrics_data: List[Dict[str, Any]], filepath: str) -> str:
    """Exports a list of metric dictionaries to a CSV file."""
    if not metrics_data:
        return filepath

    keys = metrics_data[0].keys()
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics_data)

    return filepath


def create_zip_archive(file_paths: List[str], zip_output_path: str) -> str:
    """Bundles specified output files into a ZIP archive."""
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_paths:
            if os.path.exists(file):
                zipf.write(file, os.path.basename(file))
    return zip_output_path


def generate_pdf_report(
    output_pdf_path: str,
    title: str = "Digital Image Processing Technical Report",
    author: str = "DIP-Vision.AI Team",
    original_img: Optional[np.ndarray] = None,
    processed_img: Optional[np.ndarray] = None,
    pipeline_steps: Optional[List[Dict[str, Any]]] = None,
    metrics: Optional[Dict[str, float]] = None,
    additional_notes: str = ""
) -> str:
    """
    Generates a professional 5-10 page PDF Technical Report using ReportLab.
    Includes Executive Summary, Mathematical Foundations, Algorithmic Pipeline,
    Quantitative Metrics Table, Visual Figures, and Engineering Discussion.
    """
    if not HAS_REPORTLAB:
        raise RuntimeError(
            "ReportLab package is not installed. Please run: pip install reportlab"
        )

    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=54, leftMargin=54,
        topMargin=54, bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography & styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#4F46E5"),
        spaceAfter=12
    )

    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=14,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8
    )

    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Code'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        backColor=colors.HexColor("#F1F5F9"),
        borderColor=colors.HexColor("#CBD5E1"),
        borderWidth=1,
        borderPadding=6,
        spaceAfter=10
    )

    story = []

    # Title & Metadata Header
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(f"<b>Author / Team:</b> {author} | <b>Date:</b> July 04, 2026", body_style))
    story.append(Paragraph("<b>Course:</b> DIP-Vision.AI | Digital Image Processing Toolkit", body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4F46E5"), spaceAfter=15))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", h2_style))
    exec_summary_text = (
        "This technical report documents the algorithmic implementation, visual outcomes, "
        "and quantitative metrics produced by the modular <b>Digital Image Processing Toolkit (DIP Toolkit)</b>. "
        "The toolkit encompasses 15 core processing domains including spatial transforms, histogram processing, "
        "frequency domain 2D FFT filtering, noise simulation & deblurring restoration, color space manipulation, "
        "morphological operations, segmentation, feature extraction, compression, and objective quality evaluation."
    )
    story.append(Paragraph(exec_summary_text, body_style))
    story.append(Spacer(1, 10))

    # 2. Visual Results Comparison
    if original_img is not None and processed_img is not None:
        story.append(Paragraph("2. Visual Comparison & Processing Results", h2_style))

        # Save images temporarily for ReportLab inclusion
        temp_dir = os.path.join(os.path.dirname(output_pdf_path), "temp_pdf_imgs")
        os.makedirs(temp_dir, exist_ok=True)

        orig_file = os.path.join(temp_dir, "orig.png")
        proc_file = os.path.join(temp_dir, "proc.png")

        Image.fromarray(original_img).save(orig_file)
        Image.fromarray(processed_img).save(proc_file)

        img_table_data = [[
            Paragraph("<b>Original Image</b>", body_style),
            Paragraph("<b>Processed Image</b>", body_style)
        ], [
            RLImage(orig_file, width=3.2*inch, height=2.4*inch),
            RLImage(proc_file, width=3.2*inch, height=2.4*inch)
        ]]

        t = Table(img_table_data, colWidths=[3.5*inch, 3.5*inch])
        t.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))

    # 3. Quantitative Quality Metrics
    story.append(Paragraph("3. Quantitative Image Quality Assessment (IQA)", h2_style))
    if metrics:
        metric_table_data = [["Metric", "Value", "Interpretation"]]
        descriptions = {
            "MSE": "Mean Squared Error (Lower is closer to original)",
            "RMSE": "Root Mean Squared Error (L2 Norm intensity deviation)",
            "PSNR (dB)": "Peak Signal-to-Noise Ratio (Higher indicates superior fidelity)",
            "SSIM": "Structural Similarity Index (1.0 represents perfect structural match)"
        }
        for k, v in metrics.items():
            metric_table_data.append([k, str(v), descriptions.get(k, "Fidelity metric")])

        mt = Table(metric_table_data, colWidths=[1.8*inch, 1.2*inch, 4.0*inch])
        mt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E293B")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        story.append(mt)
    else:
        story.append(Paragraph("No quantitative metrics calculated for current session.", body_style))

    story.append(Spacer(1, 15))

    # 4. Processing Pipeline Steps
    story.append(Paragraph("4. Applied Processing Pipeline", h2_style))
    if pipeline_steps:
        pipe_table_data = [["Step #", "Operation Domain", "Parameters"]]
        for i, step in enumerate(pipeline_steps):
            op = step.get("operation", "Unknown")
            params = str(step.get("params", {}))
            pipe_table_data.append([str(i + 1), op, params])

        pt = Table(pipe_table_data, colWidths=[0.8*inch, 2.2*inch, 4.0*inch])
        pt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4F46E5")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ]))
        story.append(pt)
    else:
        story.append(Paragraph("Single step operation executed.", body_style))

    story.append(Spacer(1, 15))

    # 5. Mathematical Foundations
    story.append(Paragraph("5. Mathematical Foundations & Formulations", h2_style))
    math_text = (
        "<b>2D Fast Fourier Transform (FFT):</b><br/>"
        "F(u, v) = ∑<sub>x=0</sub><sup>M-1</sup> ∑<sub>y=0</sub><sup>N-1</sup> f(x, y) e<sup>-j 2π (ux/M + vy/N)</sup><br/><br/>"
        "<b>Wiener Deconvolution Filter:</b><br/>"
        "F̂(u, v) = [ |H(u, v)|<sup>2</sup> / (|H(u, v)|<sup>2</sup> + K) ] · [ G(u, v) / H(u, v) ]<br/><br/>"
        "<b>Structural Similarity Index (SSIM):</b><br/>"
        "SSIM(x, y) = [ (2μ<sub>x</sub>μ<sub>y</sub> + C<sub>1</sub>)(2σ<sub>xy</sub> + C<sub>2</sub>) ] / "
        "[ (μ<sub>x</sub><sup>2</sup> + μ<sub>y</sub><sup>2</sup> + C<sub>1</sub>)(σ<sub>x</sub><sup>2</sup> + σ<sub>y</sub><sup>2</sup> + C<sub>2</sub>) ]"
    )
    story.append(Paragraph(math_text, code_style))

    if additional_notes:
        story.append(Spacer(1, 10))
        story.append(Paragraph("6. Additional Engineering Notes", h2_style))
        story.append(Paragraph(additional_notes, body_style))

    # Build PDF Document
    doc.build(story)
    return output_pdf_path
