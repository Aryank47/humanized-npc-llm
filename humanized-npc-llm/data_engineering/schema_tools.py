import json, jsonschema, pathlib, logging

logger = logging.getLogger(__name__)


def load_schema(path="schema/npc_schema.json"):
    base = pathlib.Path(__file__).parent
    return json.loads((base / path).read_text())


def validate_record(schema, rec):
    try:
        jsonschema.validate(instance=rec, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        logger.error(f"[schema] {rec.get('id','UNKNOWN')} failed: {e.message}")
        return False
