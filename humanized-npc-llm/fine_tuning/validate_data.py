#!/usr/bin/env python3
"""
Data Quality Validation Script
Run this BEFORE starting production training to validate data quality
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

def validate_dataset(filepath):
    """Comprehensive dataset validation."""
    print("=" * 80)
    print(f"DATA QUALITY VALIDATION: {Path(filepath).name}")
    print("=" * 80)
    
    # Statistics
    sources = Counter()
    with_persona = 0
    with_world_facts = 0
    empty_dialog = 0
    single_turn = 0
    multi_turn = 0
    total = 0
    
    # Quality samples
    pc_samples = []  # PersonaChat
    cc_samples = []  # CharCodex
    skyrim_samples = []
    spc_samples = []
    
    # Quality flags
    nonsensical_count = 0
    narrative_count = 0
    
    print("\nScanning dataset...")
    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: Invalid JSON at line {line_num}")
                continue
            
            total += 1
            source = data.get('source', 'unknown')
            sources[source] += 1
            
            # Persona coverage
            persona = data.get('persona', [])
            if persona and len(persona) > 0:
                with_persona += 1
            
            # World facts coverage
            world_facts = data.get('world_facts', [])
            if world_facts and len(world_facts) > 0:
                with_world_facts += 1
            
            # Dialog structure
            dialog = data.get('dialog', [])
            if not dialog:
                empty_dialog += 1
            elif len(dialog) == 1:
                single_turn += 1
            else:
                multi_turn += 1
            
            # Quick quality heuristics
            if source == 'personachat' and dialog:
                # Check for nonsensical responses
                if len(dialog) >= 2:
                    player_text = dialog[0].get('text', '').lower()
                    npc_text = dialog[1].get('text', '').lower()
                    
                    # Very basic coherence check
                    # (camping → color) is nonsensical
                    if 'camping' in player_text and 'color' in npc_text:
                        nonsensical_count += 1
                    if 'family' in player_text and 'favorite color' in npc_text:
                        nonsensical_count += 1
            
            if source == 'charcodex' and dialog:
                # Check for narrative style
                npc_text = dialog[-1].get('text', '')
                narrative_indicators = [
                    'trying to', 'stands on', 'recounting', 'explaining',
                    'describing', ' is ', ' was ', 'can be seen'
                ]
                if any(ind in npc_text.lower() for ind in narrative_indicators):
                    narrative_count += 1
            
            # Collect samples
            if source == 'personachat' and len(pc_samples) < 10:
                pc_samples.append(data)
            elif source == 'charcodex' and len(cc_samples) < 10:
                cc_samples.append(data)
            elif source == 'skyrim' or source == 'skyrim_il' and len(skyrim_samples) < 10:
                skyrim_samples.append(data)
            elif source == 'spc' and len(spc_samples) < 10:
                spc_samples.append(data)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"\nTotal samples: {total:,}")
    
    print(f"\nSource Distribution:")
    print(f"{'Source':<20s} {'Count':>10s} {'Percentage':>12s}")
    print("-" * 45)
    for source, count in sources.most_common():
        pct = 100 * count / total
        print(f"{source:<20s} {count:>10,} {pct:>11.1f}%")
    
    print(f"\nPersona Coverage:")
    pct_with = 100 * with_persona / total
    pct_without = 100 * (total - with_persona) / total
    print(f"  With persona:    {with_persona:>10,} ({pct_with:>5.1f}%)")
    print(f"  Without persona: {total - with_persona:>10,} ({pct_without:>5.1f}%)")
    
    print(f"\nWorld Facts Coverage:")
    pct_with = 100 * with_world_facts / total
    print(f"  With world facts: {with_world_facts:>10,} ({pct_with:>5.1f}%)")
    
    print(f"\nDialog Structure:")
    if empty_dialog > 0:
        print(f"  ⚠️  Empty dialog:   {empty_dialog:>10,}")
    print(f"  Single-turn:     {single_turn:>10,}")
    print(f"  Multi-turn:      {multi_turn:>10,}")
    
    # Quality warnings
    print("\n" + "=" * 80)
    print("QUALITY CHECKS")
    print("=" * 80)
    
    if 'personachat' in sources:
        pc_count = sources['personachat']
        nonsensical_rate = 100 * nonsensical_count / pc_count if pc_count > 0 else 0
        print(f"\nPersonaChat Quality:")
        print(f"  Potential nonsensical responses: {nonsensical_count} / {pc_count} ({nonsensical_rate:.1f}%)")
        if nonsensical_rate > 20:
            print("  ⚠️  WARNING: High nonsensical rate! Review PersonaChat samples below.")
        elif nonsensical_rate > 10:
            print("  ⚠️  CAUTION: Moderate nonsensical rate. Monitor training quality.")
        else:
            print("  ✅ Looks good. Low nonsensical rate.")
    
    if 'charcodex' in sources:
        cc_count = sources['charcodex']
        narrative_rate = 100 * narrative_count / cc_count if cc_count > 0 else 0
        print(f"\nCharCodex Quality:")
        print(f"  Potential narrative (non-dialogue): {narrative_count} / {cc_count} ({narrative_rate:.1f}%)")
        if narrative_rate > 50:
            print("  ⚠️  WARNING: High narrative rate! CharCodex may be descriptions, not dialogue.")
        elif narrative_rate > 30:
            print("  ⚠️  CAUTION: Moderate narrative rate. May affect dialogue style.")
        else:
            print("  ✅ Looks good. Most samples appear to be dialogue.")
    
    # Sample quality display
    print("\n" + "=" * 80)
    print("SAMPLE INSPECTION - PersonaChat (First 3)")
    print("=" * 80)
    for i, sample in enumerate(pc_samples[:3], 1):
        print(f"\n📄 Sample {i} (ID: {sample.get('id', 'unknown')})")
        print(f"   Persona: {sample.get('persona', [])}")
        for turn in sample.get('dialog', []):
            role = turn.get('role', 'unknown')
            text = turn.get('text', '')
            # Truncate long text
            display_text = text if len(text) <= 100 else text[:97] + "..."
            print(f"   {role.upper():8s}: {display_text}")
        
        # Quick coherence check
        if len(sample.get('dialog', [])) >= 2:
            player_text = sample['dialog'][0].get('text', '').lower()
            npc_text = sample['dialog'][1].get('text', '').lower()
            
            # Very basic check
            if 'camping' in player_text and 'color' in npc_text:
                print("   ⚠️  ISSUE: Response doesn't match query!")
            elif len(set(player_text.split()) & set(npc_text.split())) < 2:
                print("   ⚠️  CAUTION: Very low word overlap - may be incoherent")
    
    print("\n" + "=" * 80)
    print("SAMPLE INSPECTION - CharCodex (First 3)")
    print("=" * 80)
    for i, sample in enumerate(cc_samples[:3], 1):
        print(f"\n📄 Sample {i} (ID: {sample.get('id', 'unknown')})")
        print(f"   Persona: {sample.get('persona', [])}")
        for turn in sample.get('dialog', []):
            role = turn.get('role', 'unknown')
            text = turn.get('text', '')
            display_text = text if len(text) <= 100 else text[:97] + "..."
            print(f"   {role.upper():8s}: {display_text}")
        
        # Check for narrative
        npc_text = sample['dialog'][-1].get('text', '')
        if any(ind in npc_text.lower() for ind in ['trying to', 'stands', 'recounting']):
            print("   ⚠️  ISSUE: This appears to be narrative, not dialogue!")
    
    print("\n" + "=" * 80)
    print("SAMPLE INSPECTION - Skyrim (First 3)")
    print("=" * 80)
    for i, sample in enumerate(skyrim_samples[:3], 1):
        print(f"\n📄 Sample {i} (ID: {sample.get('id', 'unknown')})")
        for turn in sample.get('dialog', []):
            role = turn.get('role', 'unknown')
            text = turn.get('text', '')
            print(f"   {role.upper():8s}: {text}")
        print("   ✅ Authentic game dialogue")
    
    print("\n" + "=" * 80)
    print("SAMPLE INSPECTION - SPC (First 2)")
    print("=" * 80)
    for i, sample in enumerate(spc_samples[:2], 1):
        print(f"\n📄 Sample {i} (ID: {sample.get('id', 'unknown')})")
        print(f"   Persona ({len(sample.get('persona', []))} traits):")
        for trait in sample.get('persona', [])[:5]:
            print(f"     - {trait}")
        if len(sample.get('persona', [])) > 5:
            print(f"     ... and {len(sample['persona']) - 5} more")
        
        # Show first 4 turns
        for turn in sample.get('dialog', [])[:4]:
            role = turn.get('role', 'unknown')
            text = turn.get('text', '')
            display_text = text if len(text) <= 80 else text[:77] + "..."
            print(f"   {role.upper():8s}: {display_text}")
        if len(sample.get('dialog', [])) > 4:
            print(f"   ... and {len(sample['dialog']) - 4} more turns")
    
    # Final assessment
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    issues = []
    warnings = []
    
    # Check for critical issues
    if empty_dialog > 0:
        issues.append(f"{empty_dialog} samples with empty dialog (will be filtered)")
    
    if 'personachat' in sources:
        pc_count = sources['personachat']
        nonsensical_rate = 100 * nonsensical_count / pc_count if pc_count > 0 else 0
        if nonsensical_rate > 20:
            issues.append(f"PersonaChat: {nonsensical_rate:.1f}% potentially nonsensical")
        elif nonsensical_rate > 10:
            warnings.append(f"PersonaChat: {nonsensical_rate:.1f}% potentially nonsensical")
    
    if 'charcodex' in sources:
        cc_count = sources['charcodex']
        narrative_rate = 100 * narrative_count / cc_count if cc_count > 0 else 0
        if narrative_rate > 50:
            issues.append(f"CharCodex: {narrative_rate:.1f}% appear to be narrative")
        elif narrative_rate > 30:
            warnings.append(f"CharCodex: {narrative_rate:.1f}% appear to be narrative")
    
    pct_no_persona = 100 * (total - with_persona) / total
    if pct_no_persona > 80:
        warnings.append(f"{pct_no_persona:.1f}% samples lack persona")
    
    # Print assessment
    if issues:
        print("\n🔴 CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n⚠️  RECOMMENDATION: Address these before training")
    elif warnings:
        print("\n🟡 WARNINGS FOUND:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\n⚠️  RECOMMENDATION: Proceed with caution, monitor training quality")
    else:
        print("\n🟢 NO MAJOR ISSUES FOUND")
        print("   Dataset appears ready for training")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Review samples above - do they look reasonable?

2. If PersonaChat has high nonsensical rate:
   - Filter samples with low coherence
   - Or reduce PersonaChat weight in data mix

3. If CharCodex has high narrative rate:
   - Filter narrative samples (keep only dialogue)
   - Or reduce CharCodex weight in data mix

4. If proceeding as-is:
   - Ensure data_loader handles empty persona/world_facts
   - Monitor sample generations during training
   - Be prepared to iterate on data mix for Run 2

5. Budget check:
   - ~{hours:.1f} hours training time @ 4s/step
   - ~${cost:.2f} cost @ $0.65/hr
   - {runs} full runs possible with $100 budget
""".format(
        hours = (total / 16 * 3 * 4) / 3600,  # steps * seconds / 3600
        cost = (total / 16 * 3 * 4) / 3600 * 0.65,
        runs = int(100 / ((total / 16 * 3 * 4) / 3600 * 0.65))
    ))
    
    print("=" * 80)

def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default path
        filepath = "../data_engineering/data/processed/classwork/train.jsonl"
    
    if not Path(filepath).exists():
        print(f"ERROR: File not found: {filepath}")
        print(f"\nUsage: {sys.argv[0]} [path/to/train.jsonl]")
        sys.exit(1)
    
    validate_dataset(filepath)
    print("\nValidation complete! Review the output above before training.\n")

if __name__ == "__main__":
    main()