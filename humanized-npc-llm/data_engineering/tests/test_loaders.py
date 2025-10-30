from data_engineering.schema_tools import load_schema, validate_record
from data_engineering.loaders.persona_chat import load_persona_chat
from data_engineering.loaders.spc import load_spc
from data_engineering.loaders.character_codex import load_character_codex
from data_engineering.loaders.empathetic_dialogues import load_ed


def _check(gen):
    schema = load_schema()
    recs = [next(gen) for _ in range(3)]
    assert all(validate_record(schema, r) for r in recs)


def test_pc():
    _check(load_persona_chat("train"))


def test_spc():
    _check(load_spc("train"))


def test_cc():
    _check(load_character_codex("train"))


def test_ed():
    _check(load_ed("train"))
