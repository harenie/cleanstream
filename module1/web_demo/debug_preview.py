import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module1.scripts.run_module1_demo_server import run_module1_preview
import json

payload = {
    "question": "Evaluate the concept of ubiquity as a feature of e-commerce.",
    "student_answer": "Ubiquity means the internrt is everywhere and customers can shop anytime because the internet is available from many places, and consumers can shop all day from all locations any locations which will reduce the time and effort.",
    "model_answer": "- Ubiquity Defined: E-commerce is available anywhere and anytime through internet-connected devices.\n- Consumer Impact: Consumers can shop 24/7 from any location, reducing search time and effort.\n- Business Impact: Businesses need always-on digital infrastructure and fulfilment systems.\n- Marketspace Concept: Ubiquity removes geographic and time limits from commerce.\n- Critical Evaluation: The digital divide can exclude people without reliable internet access.",
    "processing_path": "llm",
}

result = run_module1_preview(payload)
print(json.dumps(result, indent=2, ensure_ascii=False))
