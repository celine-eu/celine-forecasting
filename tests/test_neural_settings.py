from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.models.neural_common.settings import backend_settings


def test_reads_config_section_overriding_default() -> None:
    config = load_config()
    # The yaml sets chronos_bolt covariates: false, overriding the True default.
    settings = backend_settings(config, "chronos_bolt", covariates=True)
    assert settings["covariates"] is False


def test_falls_back_to_defaults_for_unknown_backend() -> None:
    config = load_config()
    settings = backend_settings(config, "no_such_backend", context_length=99, finetune=True)
    assert settings == {"context_length": 99, "finetune": True, "covariates": True}
