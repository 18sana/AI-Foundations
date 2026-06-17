import sys
import subprocess


def test_eval_pipeline():

    result = subprocess.run(
        [
            sys.executable,
            "evals/run_eval.py"
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0