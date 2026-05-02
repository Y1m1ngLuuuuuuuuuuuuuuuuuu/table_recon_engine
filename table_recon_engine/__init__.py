__all__ = ["TSREngine"]


def __getattr__(name: str):
    if name == "TSREngine":
        from table_recon_engine.models import TSREngine

        return TSREngine
    raise AttributeError(name)
