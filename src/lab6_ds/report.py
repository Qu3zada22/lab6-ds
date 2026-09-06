"""Execute the existing notebook and render a compact, code-free PDF report."""

import argparse
import base64
import hashlib
from io import BytesIO
from pathlib import Path
import time
from xml.sax.saxutils import escape

from bs4 import BeautifulSoup
import mistune
import nbformat
from nbclient import NotebookClient
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, LongTable, TableStyle,
)

NOTEBOOK = "Laboratorio6_Redes_Sociales_YouTube.ipynb"


def render_report(notebook, target):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Cell", fontName="Helvetica", fontSize=7, leading=9, wordWrap="CJK"))
    styles["BodyText"].fontSize = 9
    styles["BodyText"].leading = 12
    story = [Paragraph("Informe de resultados — Laboratorio 6", styles["Title"]),
             Paragraph("Análisis de participación en YouTube | CC3084, UVG", styles["Heading2"]),
             Paragraph("Fuente: notebook ejecutado. Se omite el código; las tablas muy anchas o extensas "
                       "se resumen aquí y se conservan completas en el notebook y los CSV de results/. "
                       "Las etiquetas de sentimiento son predicciones no validadas localmente.", styles["BodyText"])]

    def paragraph(text, style="BodyText"):
        if text.strip():
            story.append(Paragraph(escape(text).replace("\n", "<br/>"), styles[style]))
            story.append(Spacer(1, 4))

    def html_content(html):
        soup = BeautifulSoup(html, "html.parser")
        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
            if element.find_parent(["table", "li"]):
                continue
            if element.name == "table":
                table_content(element)
            else:
                style = {"h1": "Heading1", "h2": "Heading1", "h3": "Heading2", "h4": "Heading3"}.get(element.name, "BodyText")
                text = element.get_text(" ", strip=True)
                if element.name == "li":
                    text = "• " + text
                paragraph(text, style)

    def table_content(table):
        rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                for row in table.find_all("tr")]
        if not rows:
            return
        columns = max(map(len, rows))
        if columns > 8:
            # Vertical records keep all fields legible, but bound repeated output.
            headers = rows[0]
            for row in rows[1:6]:
                paragraph(" | ".join(f"{key}: {value[:150]}" for key, value in zip(headers, row)), "Cell")
            paragraph("Tabla extensa: se presentan hasta cinco registros. Consulte la tabla completa en el notebook/CSV.", "Cell")
            return
        shown = rows[:26]
        data = [[Paragraph(escape(value[:180]), styles["Cell"]) for value in row + [""] * (columns - len(row))]
                for row in shown]
        table_object = LongTable(data, colWidths=[7 * inch / columns] * columns, repeatRows=1, hAlign="LEFT")
        table_object.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF6E8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.2, colors.lightgrey),
        ]))
        story.extend([table_object, Spacer(1, 8)])
        if len(rows) > 26:
            paragraph("Tabla resumida a 25 registros; véase notebook/CSV para todos los registros.", "Cell")

    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            html_content(mistune.html(cell.source))
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                image = Image(BytesIO(base64.b64decode(data["image/png"])))
                scale = min(7 * inch / image.imageWidth, 7 * inch / image.imageHeight)
                image.drawWidth = image.imageWidth * scale
                image.drawHeight = image.imageHeight * scale
                story.extend([image, Spacer(1, 8)])
            elif "text/markdown" in data:
                html_content(mistune.html(data["text/markdown"]))
            elif "text/html" in data:
                html_content(data["text/html"])
            else:
                text = output.get("text", data.get("text/plain", ""))
                lines = text.splitlines()
                for line in lines[:60]:
                    paragraph(line[:220], "Cell")
                if len(lines) > 60:
                    paragraph("Salida resumida; véase notebook para el detalle.", "Cell")

    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(0.6 * inch, 0.35 * inch, "Laboratorio 6 | Participación observada, no inferencia poblacional")
        canvas.drawRightString(7.9 * inch, 0.35 * inch, str(document.page))

    document = SimpleDocTemplate(str(target), pagesize=(8.5 * inch, 11 * inch),
                                 rightMargin=0.7 * inch, leftMargin=0.7 * inch,
                                 topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                                 title="Laboratorio 6 — Análisis de redes sociales YouTube",
                                 author="")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return document.page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true", help="Render an already successfully executed notebook")
    args = parser.parse_args()
    path = Path(NOTEBOOK)
    if not path.exists():
        parser.error("Run this command from the repository root.")
    notebook = nbformat.read(path, as_version=4)
    source_digest = hashlib.sha256("\n".join(cell.source for cell in notebook.cells).encode()).hexdigest()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
    start = time.perf_counter()
    if not args.report_only:
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        NotebookClient(notebook, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(Path.cwd())}},
                       allow_errors=False).execute()
        notebook.metadata["executed_source_sha256"] = source_digest
        nbformat.write(notebook, path)
    elif notebook.metadata.get("executed_source_sha256") != source_digest:
        raise RuntimeError("Notebook sources changed since execution; run without --report-only.")
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
    if any(cell.execution_count is None or any(out.output_type == "error" for out in cell.outputs) for cell in code_cells):
        raise RuntimeError("Refusing to report an incomplete or failed notebook. Execute without --report-only.")
    target = Path("results")
    target.mkdir(exist_ok=True)
    pages = render_report(notebook, target / "Informe_Laboratorio6.pdf")
    print(f"Notebook: {len(code_cells)} executed code cells; PDF: {pages} pages; elapsed: {time.perf_counter() - start:.1f}s")
