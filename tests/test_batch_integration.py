from pathlib import Path

from docking_confidence.application.batch_launcher import demo_batch_command


def test_demo_worker_command_is_explicit_and_never_uses_a_shell(tmp_path: Path):
    command = demo_batch_command(
        input_dir=tmp_path / "entrada com espaço",
        output_dir=tmp_path / "saída",
        algorithms=["pso", "ag"],
        protocol="reduced_vs_full",
        generations=100,
        population=100,
        top_pockets=32,
        seed_start=1,
        seed_count=2,
        resume_after_cancel=True,
    )

    assert command[1:4] == ("-m", "docking_confidence", "batch-run-demo")
    assert "--acknowledge-non-scientific-demo" in command
    assert "--resume-after-cancel" in command
    assert str(tmp_path / "entrada com espaço") in command
    assert str(tmp_path / "saída") in command


def test_batch_streamlit_page_imports_without_rendering_widgets():
    from docking_confidence.ui.batch_page import render_batch_page

    assert callable(render_batch_page)
