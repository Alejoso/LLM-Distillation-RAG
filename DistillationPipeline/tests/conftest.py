import importlib
import sys
import types

import pytest


@pytest.fixture
def distill_data_module(monkeypatch):
    """
    Importa src.distill_data con un src.rag_adapter fake
    para evitar dependencias pesadas en pruebas unitarias.
    """
    fake_rag_adapter = types.ModuleType("src.rag_adapter")
    fake_rag_adapter.retrieve_contexts = lambda instruction, config: []

    monkeypatch.setitem(sys.modules, "src.rag_adapter", fake_rag_adapter)
    sys.modules.pop("src.distill_data", None)

    module = importlib.import_module("src.distill_data")
    return module
