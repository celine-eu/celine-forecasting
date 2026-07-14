from typer.testing import CliRunner

from celine.forecasting.cli import app

runner = CliRunner()


def test_run_help_lists_model_and_scope() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--scope" in result.output


def test_train_help_lists_model_and_scope() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--scope" in result.output
