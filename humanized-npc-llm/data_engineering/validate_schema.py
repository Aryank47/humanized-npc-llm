#!/usr/bin/env python3
"""
JSON Schema Validation for NPC Dialogue Dataset

Validates that all train/val/test splits conform to the required schema.
Critical pre-training check to prevent data loading errors.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

# ============================================================================
# SCHEMA DEFINITION
# ============================================================================

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "NPC Dialogue Record",
    "type": "object",
    "required": ["id", "source", "split", "dialog"],
    "properties": {
        "id": {"type": "string"},
        "source": {
            "type": "string", 
            "enum": ["personachat", "spc", "charcodex", "light", "skyrim", "fallout", "ed"]
        },
        "split": {"type": "string"},
        "persona": {"type": "array", "items": {"type": "string"}},
        "world_facts": {"type": "array", "items": {"type": "string"}},
        "context": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "npc_role": {"type": "string"},
                "time_of_day": {"type": "string"},
                "player_state": {"type": "object"}
            }
        },
        "intent": {"type": "string"},
        "control": {"type": "object"},
        "dialog": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["role", "text"],
                "properties": {
                    "role": {"type": "string"},
                    "text": {"type": "string"}
                }
            }
        },
        "choices": {"type": "array", "items": {"type": "object"}},
        "transaction": {"type": "object"},
        "meta": {"type": "object"}
    },
    "additionalProperties": False
}

ALLOWED_SOURCES = ["personachat", "spc", "charcodex", "light", "skyrim", "fallout", "ed"]
REQUIRED_FIELDS = ["id", "source", "split", "dialog"]

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_type(value, expected_type: str, field_path: str) -> Tuple[bool, str]:
    """Validate that a value matches the expected type."""
    type_mapping = {
        "string": str,
        "array": list,
        "object": dict,
    }
    
    expected_python_type = type_mapping.get(expected_type)
    if expected_python_type and not isinstance(value, expected_python_type):
        return False, f"{field_path}: Expected {expected_type}, got {type(value).__name__}"
    
    return True, ""


def validate_record(record: Dict, line_num: int) -> Tuple[bool, List[str]]:
    """
    Validate a single record against the schema.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # 1. Check for additional properties
    allowed_props = set(SCHEMA["properties"].keys())
    actual_props = set(record.keys())
    extra_props = actual_props - allowed_props
    
    if extra_props:
        errors.append(f"Line {line_num}: Unexpected properties: {extra_props}")
    
    # 2. Check required fields
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"Line {line_num}: Missing required field '{field}'")
    
    # 3. Validate field types and constraints
    
    # ID
    if "id" in record:
        valid, error = validate_type(record["id"], "string", f"Line {line_num}.id")
        if not valid:
            errors.append(error)
    
    # Source (with enum check)
    if "source" in record:
        valid, error = validate_type(record["source"], "string", f"Line {line_num}.source")
        if not valid:
            errors.append(error)
        elif record["source"] not in ALLOWED_SOURCES:
            errors.append(
                f"Line {line_num}.source: '{record['source']}' not in allowed values {ALLOWED_SOURCES}"
            )
    
    # Split
    if "split" in record:
        valid, error = validate_type(record["split"], "string", f"Line {line_num}.split")
        if not valid:
            errors.append(error)
    
    # Persona (optional array of strings)
    if "persona" in record:
        valid, error = validate_type(record["persona"], "array", f"Line {line_num}.persona")
        if not valid:
            errors.append(error)
        elif not all(isinstance(item, str) for item in record["persona"]):
            errors.append(f"Line {line_num}.persona: All items must be strings")
    
    # World Facts (optional array of strings)
    if "world_facts" in record:
        valid, error = validate_type(record["world_facts"], "array", f"Line {line_num}.world_facts")
        if not valid:
            errors.append(error)
        elif not all(isinstance(item, str) for item in record["world_facts"]):
            errors.append(f"Line {line_num}.world_facts: All items must be strings")
    
    # Context (optional object)
    if "context" in record:
        valid, error = validate_type(record["context"], "object", f"Line {line_num}.context")
        if not valid:
            errors.append(error)
    
    # Dialog (CRITICAL - required array with minItems: 1)
    if "dialog" in record:
        valid, error = validate_type(record["dialog"], "array", f"Line {line_num}.dialog")
        if not valid:
            errors.append(error)
        elif len(record["dialog"]) < 1:
            errors.append(f"Line {line_num}.dialog: Must have at least 1 item (minItems: 1)")
        else:
            # Validate each dialog turn
            for i, turn in enumerate(record["dialog"]):
                if not isinstance(turn, dict):
                    errors.append(f"Line {line_num}.dialog[{i}]: Must be an object")
                    continue
                
                # Check required fields: role, text
                if "role" not in turn:
                    errors.append(f"Line {line_num}.dialog[{i}]: Missing required field 'role'")
                elif not isinstance(turn["role"], str):
                    errors.append(f"Line {line_num}.dialog[{i}].role: Must be a string")
                
                if "text" not in turn:
                    errors.append(f"Line {line_num}.dialog[{i}]: Missing required field 'text'")
                elif not isinstance(turn["text"], str):
                    errors.append(f"Line {line_num}.dialog[{i}].text: Must be a string")
    
    # Choices (optional array of objects)
    if "choices" in record:
        valid, error = validate_type(record["choices"], "array", f"Line {line_num}.choices")
        if not valid:
            errors.append(error)
    
    # Transaction (optional object)
    if "transaction" in record:
        valid, error = validate_type(record["transaction"], "object", f"Line {line_num}.transaction")
        if not valid:
            errors.append(error)
    
    # Meta (optional object)
    if "meta" in record:
        valid, error = validate_type(record["meta"], "object", f"Line {line_num}.meta")
        if not valid:
            errors.append(error)
    
    is_valid = len(errors) == 0
    return is_valid, errors


# ============================================================================
# FILE VALIDATION
# ============================================================================

def validate_file(file_path: str) -> Dict:
    """
    Validate an entire JSONL file.
    
    Returns:
        Dict with validation results and statistics
    """
    print(f"\n{'=' * 80}")
    print(f"VALIDATING: {file_path}")
    print(f"{'=' * 80}")
    
    if not Path(file_path).exists():
        return {
            "file": file_path,
            "exists": False,
            "error": "File not found"
        }
    
    results = {
        "file": file_path,
        "exists": True,
        "total_records": 0,
        "valid_records": 0,
        "invalid_records": 0,
        "parse_errors": 0,
        "validation_errors": [],
        "source_distribution": defaultdict(int),
        "split_values": set(),
        "persona_coverage": {"with_persona": 0, "without_persona": 0},
        "dialog_stats": {"min_turns": float('inf'), "max_turns": 0, "total_turns": 0}
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            # Try to parse JSON
            try:
                record = json.loads(line)
                results["total_records"] += 1
            except json.JSONDecodeError as e:
                results["parse_errors"] += 1
                results["validation_errors"].append(f"Line {line_num}: JSON parse error: {e}")
                continue
            
            # Validate against schema
            is_valid, errors = validate_record(record, line_num)
            
            if is_valid:
                results["valid_records"] += 1
                
                # Collect statistics
                if "source" in record:
                    results["source_distribution"][record["source"]] += 1
                
                if "split" in record:
                    results["split_values"].add(record["split"])
                
                # Persona coverage
                if record.get("persona") and len(record["persona"]) > 0:
                    results["persona_coverage"]["with_persona"] += 1
                else:
                    results["persona_coverage"]["without_persona"] += 1
                
                # Dialog statistics
                if "dialog" in record:
                    num_turns = len(record["dialog"])
                    results["dialog_stats"]["min_turns"] = min(
                        results["dialog_stats"]["min_turns"], num_turns
                    )
                    results["dialog_stats"]["max_turns"] = max(
                        results["dialog_stats"]["max_turns"], num_turns
                    )
                    results["dialog_stats"]["total_turns"] += num_turns
            else:
                results["invalid_records"] += 1
                results["validation_errors"].extend(errors)
    
    return results


# ============================================================================
# REPORTING
# ============================================================================

def print_results(results: Dict):
    """Print validation results in a readable format."""
    
    if not results["exists"]:
        print(f"  ❌ ERROR: {results['error']}")
        return
    
    total = results["total_records"]
    valid = results["valid_records"]
    invalid = results["invalid_records"]
    parse_errors = results["parse_errors"]
    
    print(f"\nRecords Summary:")
    print(f"  Total records: {total:,}")
    print(f"  Valid records: {valid:,} ({valid/total*100:.1f}%)")
    print(f"  Invalid records: {invalid:,}")
    print(f"  Parse errors: {parse_errors}")
    
    # Overall status
    print(f"\nOverall Status:")
    if invalid == 0 and parse_errors == 0:
        print(f"  ✅ ALL RECORDS VALID")
    else:
        print(f"  ❌ VALIDATION FAILED")
    
    # Source distribution
    if results["source_distribution"]:
        print(f"\nSource Distribution:")
        for source, count in sorted(results["source_distribution"].items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            print(f"    {source}: {count:,} ({pct:.1f}%)")
    
    # Split values
    if results["split_values"]:
        print(f"\nSplit Values Found: {', '.join(sorted(results['split_values']))}")
    
    # Persona coverage
    with_persona = results["persona_coverage"]["with_persona"]
    without_persona = results["persona_coverage"]["without_persona"]
    persona_pct = with_persona / total * 100 if total > 0 else 0
    
    print(f"\nPersona Coverage:")
    print(f"  With persona: {with_persona:,} ({persona_pct:.1f}%)")
    print(f"  Without persona: {without_persona:,}")
    
    if persona_pct >= 95:
        print(f"  ✅ EXCELLENT persona coverage")
    elif persona_pct >= 40:
        print(f"  ⚠️  ACCEPTABLE persona coverage")
    else:
        print(f"  ❌ LOW persona coverage")
    
    # Dialog statistics
    if results["dialog_stats"]["total_turns"] > 0:
        avg_turns = results["dialog_stats"]["total_turns"] / total
        print(f"\nDialog Statistics:")
        print(f"  Min turns: {results['dialog_stats']['min_turns']}")
        print(f"  Max turns: {results['dialog_stats']['max_turns']}")
        print(f"  Avg turns: {avg_turns:.1f}")
    
    # Show first few errors (if any)
    if results["validation_errors"]:
        print(f"\n❌ Validation Errors (showing first 20):")
        for error in results["validation_errors"][:20]:
            print(f"  {error}")
        
        if len(results["validation_errors"]) > 20:
            print(f"  ... and {len(results['validation_errors']) - 20} more errors")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Validate all three splits."""
    
    print("=" * 80)
    print("JSON SCHEMA VALIDATION")
    print("NPC Dialogue Dataset - Final Pre-Training Check")
    print("=" * 80)
    
    # File paths
    base_dir = Path("data/processed/v2")
    splits = ["train", "val", "test"]
    
    # Validate each split
    all_results = {}
    all_valid = True
    
    for split in splits:
        file_path = base_dir / f"{split}.jsonl"
        results = validate_file(str(file_path))
        all_results[split] = results
        print_results(results)
        
        if results.get("invalid_records", 0) > 0 or results.get("parse_errors", 0) > 0:
            all_valid = False
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    
    total_records = sum(r.get("total_records", 0) for r in all_results.values())
    total_valid = sum(r.get("valid_records", 0) for r in all_results.values())
    total_invalid = sum(r.get("invalid_records", 0) for r in all_results.values())
    
    print(f"\nAcross All Splits:")
    print(f"  Total records: {total_records:,}")
    print(f"  Valid records: {total_valid:,}")
    print(f"  Invalid records: {total_invalid:,}")
    
    if all_valid:
        print(f"\n✅ ALL SPLITS PASS SCHEMA VALIDATION")
        print(f"\nReady for training!")
        print(f"\nNext steps:")
        print(f"1. Upload to Azure Storage")
        print(f"2. Update training config with v2 data paths")
        print(f"3. Start training")
        return 0
    else:
        print(f"\n❌ SCHEMA VALIDATION FAILED")
        print(f"\nPlease fix the errors above before training.")
        print(f"\nCommon issues:")
        print(f"  - Additional properties not in schema")
        print(f"  - Invalid source enum values")
        print(f"  - Missing required fields (id, source, split, dialog)")
        print(f"  - Dialog array is empty (must have minItems: 1)")
        print(f"  - Dialog turns missing 'role' or 'text' fields")
        return 1


if __name__ == "__main__":
    sys.exit(main())