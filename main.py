from pathlib import Path
import sys

src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rag_api.main import app


def main() -> None:
    print("Run with: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
