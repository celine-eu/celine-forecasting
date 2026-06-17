from celine.meter_forecasting.core.config import load_config
from celine.meter_forecasting.models.ttm.config import TTM_MODEL_ID, ttm_settings


def test_model_id() -> None:
    assert TTM_MODEL_ID == "ibm-granite/granite-timeseries-ttm-r2"


def test_settings_defaults() -> None:
    s = ttm_settings(load_config())
    assert s["context_length"] == 512
    assert isinstance(s["finetune"], bool)
    assert isinstance(s["covariates"], bool)
