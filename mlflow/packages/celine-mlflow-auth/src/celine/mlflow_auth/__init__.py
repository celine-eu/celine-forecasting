def authorize_request():
    from celine.mlflow_auth.auth import authorize_request as _impl

    return _impl()


def create_app(app=None):
    from celine.mlflow_auth.auth import create_app as _impl

    return _impl(app)


__all__ = ["authorize_request", "create_app"]
