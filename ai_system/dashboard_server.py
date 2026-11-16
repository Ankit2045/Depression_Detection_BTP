"""
Interactive Clinical AI Dashboard Server (DTU-GP178).

Combines:
1. Static web server for the Clinician Web Dashboard (ai_system/dashboard/).
2. Interactive API endpoint (/api/predict) allowing psychiatrists and examiners to:
   - Upload any raw actigraphy CSV directly from the web browser.
   - Run multimodal deep learning inference (Dual-Fusion Ensemble).
   - Generate DSM-5 clinical consultation memos in real-time.
   - Instantly render interactive charts (waveforms, attention weights, Z-scores).
3. Sample data endpoint (/api/sample) for 1-click viva testing.
"""

import os
import sys
import json
import logging
import tempfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ai_system.predict_new_patient import NewPatientPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "dashboard"))
predictor = None


class ClinicalDashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/sample":
            # Return sample actigraphy data for 1-click viva testing
            params = urllib.parse.parse_qs(parsed.query)
            sample_type = params.get("type", ["condition"])[0].lower()
            
            sample_map = {
                "retarded": ("data/test/test_cond_retarded_01.csv", "Demo_Patient_MelancholicRetarded", "45-49", "Female"),
                "agitated": ("data/test/test_cond_agitated_01.csv", "Demo_Patient_AgitatedMixed", "30-34", "Female"),
                "atypical": ("data/test/test_cond_atypical_01.csv", "Demo_Patient_AtypicalPhaseDelayed", "20-24", "Female"),
                "active": ("data/test/test_ctrl_active_01.csv", "Demo_Patient_HealthyActive", "30-34", "Male"),
                "sedentary": ("data/test/test_ctrl_sedentary_02.csv", "Demo_Patient_HealthySedentary", "25-29", "Female"),
                "condition": ("data/test/test_cond_retarded_01.csv", "Demo_Patient_DepressionCondition", "35-39", "Female"),
                "control": ("data/test/test_ctrl_active_01.csv", "Demo_Patient_HealthyControl", "45-49", "Male"),
            }

            if sample_type in sample_map:
                sample_file, s_id, s_age, s_gender = sample_map[sample_type]
            else:
                sample_file = "data/condition/condition_1.csv"
                s_id = f"Demo_Patient_{sample_type.capitalize()}"
                s_age = "35-39"
                s_gender = "Female"

            if not os.path.exists(sample_file):
                self.send_error(404, f"Sample file {sample_file} not found")
                return

            with open(sample_file, "r") as f:
                # Read 5,000 minutes (more than 3 full days of continuous data)
                lines = [f.readline() for _ in range(5000)]
                content = "".join([l for l in lines if l])

            response = {
                "sample_type": sample_type,
                "suggested_id": s_id,
                "suggested_age": s_age,
                "suggested_gender": s_gender,
                "csv_content": content,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                data = json.loads(body)

                csv_content = data.get("csv_content", "")
                patient_id = data.get("patient_id", "new_patient_01").strip().replace(" ", "_")
                age = data.get("age", "30-34")
                gender = data.get("gender", "Female")

                if not csv_content:
                    self.send_error(400, "Missing csv_content")
                    return

                # Write to temp file for predictor
                with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
                    tmp.write(csv_content)
                    tmp_path = tmp.name

                global predictor
                if predictor is None:
                    predictor = NewPatientPredictor()

                res = predictor.predict_patient(
                    file_path=tmp_path,
                    patient_id=patient_id,
                    age=age,
                    gender=gender,
                    update_dashboard=True,
                )

                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "result": res}).encode("utf-8"))
                return

            except Exception as e:
                logger.error(f"Error processing prediction request: {e}", exc_info=True)
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
                return

        self.send_error(404, "Endpoint not found")


def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, ClinicalDashboardHandler)
    logger.info(f"Clinical AI Dashboard & Inference Server running at http://localhost:{port}/")
    print("\n" + "=" * 80)
    print(f" CLINICAL AI DECISION SUPPORT DASHBOARD LIVE")
    print(f" Open your browser: http://localhost:{port}")
    print(f" Features: Live 55-Patient Selector, Interactive Waveforms, and Direct CSV Testing")
    print("=" * 80 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping dashboard server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8000))
    run_server(port)
