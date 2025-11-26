import csv
import json
import os
import re
import sys


def create_record(id_suffix: str, source_file: str, dialog_list: list) -> dict:
    """
    Creates a dialogue record dictionary that conforms to the target JSON schema.

    All Elder Scrolls data sources are mapped to "skyrim" to satisfy
    the schema's 'source' enum.
    """

    # Get a simple prefix from the filename, e.g., "morrowwind"
    source_prefix = os.path.basename(source_file).split(".")[0].split("_")[0].lower()

    # Create a unique ID
    record_id = f"{source_prefix}_{id_suffix}"

    # Ensure dialog_list is valid
    if not dialog_list or not all(
        isinstance(d, dict) and "role" in d and "text" in d for d in dialog_list
    ):
        return None

    record = {
        "id": record_id,
        "source": "skyrim",  # Mapping all to 'skyrim' to fit the schema's enum
        "split": "train",
        "persona": [],
        "world_facts": [],
        "context": {
            "npc_role": "Skyrim NPC"  # Using this as a generic stand-in for "Elder Scrolls NPC"
        },
        "intent": "generic",
        "control": {"style": ["fantasy"]},
        "dialog": dialog_list,
        "choices": [],
        "transaction": {},
        "meta": {
            "license": "unknown",
            "dataset_ref": "local_file",
            "file": source_file,
        },
    }
    return record


def clean_text(text: str) -> str:
    """
    Removes common game placeholders and extra whitespace.
    """
    if not text:
        return ""
    # Remove placeholders like %t, %rf, %mm, %i, %en, %cn, %st, %cn2 etc.
    text = re.sub(r"%\w+", "", text)
    # Condense all whitespace (newlines, tabs, spaces) into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_oblivion(filename: str):
    """
    Parsers oblivion.txt
    Format: FormID: 00092D86\tMQ02\tMQ02JauffreD2\t0\tHis meaning is... \t[optional_tone]
    """
    print(f"Parsing {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.startswith("FormID:"):
                continue

            try:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue

                form_id = parts[0].replace("FormID: ", "").strip()
                dialog_id = parts[2]
                line_index = parts[3]
                text = parts[4]

                # Filter out metadata lines like ((attack grunt...))
                if text.startswith("((") and text.endswith("))"):
                    continue

                # Check for optional tone/emotion
                if len(parts) > 5 and parts[5]:
                    tone = parts[5].strip()
                    text = f"({tone}) {text}"  # Prepend tone to text

                text = clean_text(text)
                if not text:
                    continue

                dialog_list = [{"role": "npc", "text": text}]
                record = create_record(
                    id_suffix=f"{form_id}_{dialog_id}_{line_index}",
                    source_file=filename,
                    dialog_list=dialog_list,
                )
                if record:
                    yield record
            except Exception as e:
                print(f"Error parsing {filename} line {i+1}: {e}", file=sys.stderr)


def parse_morrowwind(filename: str):
    """
    Parsers morrowwind.txt
    Format: TSV file with quotes. We are interested in "Topic" rows.
    Example: "..."\t"..."\t"..."\t"Topic"\t"duties"\t"Ajira knows..."\t...
    """
    print(f"Parsing {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        for i, row in enumerate(reader):
            try:
                if len(row) > 5 and row[3] == "Topic":
                    text = clean_text(row[5])
                    if not text:
                        continue

                    dialog_list = [{"role": "npc", "text": text}]
                    # Use row[2] as it looks like a unique entry ID
                    record_id = row[2] if row[2] else f"line_{i+1}"

                    record = create_record(
                        id_suffix=record_id,
                        source_file=filename,
                        dialog_list=dialog_list,
                    )
                    if record:
                        yield record
            except Exception as e:
                print(f"Error parsing {filename} line {i+1}: {e}", file=sys.stderr)


def parse_battlespire(filename: str):
    """
    Parsers battlespire.txt
    Format: Tab-separated, complex structure.
    Line: \tID\tNPC_Text\t...\t[PLAYER_Response_Text]\t...
    """
    print(f"Parsing {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.startswith("\t"):
                continue

            try:
                # Skip header row
                if "NPC SAY" in line and "Replycode" in line:
                    continue

                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue

                # File is inconsistent: col 1 and 2 are swapped at times.
                # Heuristic: ID is short and/or all-caps, text is long and has spaces.
                col1 = parts[1]
                col2 = parts[2].strip("[]")

                dialog_id = ""
                npc_text_raw = ""

                if (len(col1) > len(col2) and " " in col1) or (
                    col2.isupper() and len(col2) < 15 and col1 != "NPC SAY"
                ):
                    # Case 1: col1 is text, col2 is ID (matches your bad samples)
                    dialog_id = col2
                    npc_text_raw = col1
                else:
                    # Case 2: col1 is ID, col2 is text (matches file snippet)
                    dialog_id = col1
                    npc_text_raw = col2

                npc_text = clean_text(npc_text_raw)

                if not npc_text or not dialog_id:
                    continue

                dialog_list = [{"role": "npc", "text": npc_text}]

                # Check for a player response in the 7th column (index 6)
                if len(parts) > 6 and parts[6]:
                    player_text = clean_text(parts[6])
                    if player_text:
                        dialog_list.append({"role": "player", "text": player_text})

                record = create_record(
                    id_suffix=dialog_id, source_file=filename, dialog_list=dialog_list
                )
                if record:
                    yield record
            except Exception as e:
                print(f"Error parsing {filename} line {i+1}: {e}", file=sys.stderr)


def parse_dagerfall(filename: str):
    """
    Parsers dagerfall.txt
    Format: Descriptions and questions, seems to be latin-1 encoded.
    We skip the multi-line questions as they don't fit the schema well.
    """
    print(f"Parsing {filename}...")
    with open(filename, "r", encoding="latin-1") as f:
        for i, line in enumerate(f):
            try:
                line = line.strip()
                if (
                    not line
                    or line.startswith("{")
                    or "a)" in line
                    or "b)" in line
                    or "c)" in line
                ):
                    continue

                # Fix for b'...' and escaped byte literals in the text file
                if line.startswith("b'") and line.endswith("'"):
                    line = line[2:-1]
                if line.startswith('b"') and line.endswith('"'):
                    line = line[2:-1]

                # Un-escape sequences like \\xfd
                try:
                    # This handles \\xfd -> \xfd -> (char)
                    line = line.encode("latin-1", "ignore").decode("unicode_escape")
                except Exception:
                    pass  # Ignore if unicode_escape fails

                text = clean_text(line)

                if not text or len(text.split()) < 3:  # Skip very short lines
                    continue

                dialog_list = [
                    {"role": "npc", "text": text}
                ]  # Treat descriptions as NPC narration
                record = create_record(
                    id_suffix=f"line_{i+1}",
                    source_file=filename,
                    dialog_list=dialog_list,
                )
                if record:
                    yield record
            except Exception as e:
                print(f"Error parsing {filename} line {i+1}: {e}", file=sys.stderr)


def parse_arena_city_greetings(filename: str):
    """
    Parsers arena_city_greetings.txt
    Format: [Location]\nGreeting text...\n\n[Location2]...
    """
    print(f"Parsing {filename}...")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = content.split("\n\n")
        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")
            if not lines[0].startswith("["):
                continue

            location = lines[0].strip("[]")
            text = clean_text(" ".join(lines[1:]))

            if not text:
                continue

            dialog_list = [{"role": "npc", "text": text}]
            record = create_record(
                id_suffix=f"greeting_{location.lower().replace(' ', '_')}",
                source_file=filename,
                dialog_list=dialog_list,
            )
            if record:
                yield record
    except Exception as e:
        print(f"Error parsing {filename}: {e}", file=sys.stderr)


def parse_arena_obj(filename: str):
    """
    Parsers arena_obj.txt
    Format: Title\nDescription text...\n#\nTitle2...
    """
    print(f"Parsing {filename}...")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = content.split("\n#\n")
        for i, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue

            lines = block.split("\n")
            if len(lines) < 2:
                continue

            title = lines[0].strip()
            text = clean_text(" ".join(lines[1:]))

            if not text:
                continue

            dialog_list = [{"role": "npc", "text": text}]
            record = create_record(
                id_suffix=f"obj_{title.lower().replace(' ', '_')}",
                source_file=filename,
                dialog_list=dialog_list,
            )
            if record:
                yield record
    except Exception as e:
        print(f"Error parsing {filename}: {e}", file=sys.stderr)


def parse_arena_other_dialogues(filename: str):
    """
    Parsers arena_other_dialouges.txt
    Format: #ID\nText...&\n\nText...&\n\n
    """
    print(f"Parsing {filename}...")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by the main delimiter
        blocks = content.split("&\n")
        for i, block in enumerate(blocks):
            # Each block might have sub-blocks separated by \n\n
            sub_blocks = block.split("\n\n")
            for j, sub_block in enumerate(sub_blocks):
                sub_block = sub_block.strip()
                if not sub_block:
                    continue

                lines = sub_block.split("\n")
                id_suffix = f"other_{i}_{j}"

                if lines[0].startswith("#"):
                    id_suffix = f"other_{lines[0].strip().replace(' ', '_')}"
                    text = clean_text(" ".join(lines[1:]))
                else:
                    text = clean_text(" ".join(lines))

                if not text:
                    continue

                dialog_list = [{"role": "npc", "text": text}]
                record = create_record(
                    id_suffix=id_suffix, source_file=filename, dialog_list=dialog_list
                )
                if record:
                    yield record
    except Exception as e:
        print(f"Error parsing {filename}: {e}", file=sys.stderr)


def parse_arena_snippets(filename: str):
    """
    Generic parser for Arena_EQUIP.txt, Arena_MUGUILD.txt, Arena_SELLING.txt,
    and Arena_TAVERN.txt
    Format: Snippets separated by '...' or '? ? ? ? ? ?'
    """
    print(f"Parsing {filename}...")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()

        # Normalize delimiters
        content = content.replace("? ? ? ? ? ?", "...")
        content = re.sub(
            r"&\n", "...", content
        )  # From other_dialogues, but might apply
        content = re.sub(r";.*", "...", content)  # Treat comments as delimiters

        snippets = content.split("...")
        for i, snippet in enumerate(snippets):
            text = clean_text(snippet)
            if not text:
                continue

            dialog_list = [{"role": "npc", "text": text}]
            record = create_record(
                id_suffix=f"snippet_{i+1}",
                source_file=filename,
                dialog_list=dialog_list,
            )
            if record:
                yield record
    except Exception as e:
        print(f"Error parsing {filename}: {e}", file=sys.stderr)


def parse_skyrim(filename: str):
    """
    Parsers skyrim.txt
    Format: FormID: 0401AAEF\tDLC1VQ07Post\t...\t0\tI feel nothing... \t[optional_tone]
    This format is identical to oblivion.txt, so we reuse that logic.
    """
    print(f"Parsing {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.startswith("FormID:"):
                continue

            try:
                parts = line.strip().split("\t")
                if len(parts) < 7:  # skyrim.txt seems to have 7+ columns
                    continue

                form_id = parts[0].replace("FormID: ", "").strip()
                dialog_id = parts[3]
                line_index = parts[5]
                text = parts[6]

                # Filter out metadata lines
                if text.startswith("((") and text.endswith("))"):
                    continue

                # Check for optional tone/emotion
                if len(parts) > 7 and parts[7]:
                    tone = parts[7].strip()
                    text = f"({tone}) {text}"  # Prepend tone to text

                text = clean_text(text)
                if not text:
                    continue

                dialog_list = [{"role": "npc", "text": text}]
                record = create_record(
                    id_suffix=f"{form_id}_{dialog_id}_{line_index}",
                    source_file=filename,
                    dialog_list=dialog_list,
                )
                if record:
                    yield record
            except Exception as e:
                print(f"Error parsing {filename} line {i+1}: {e}", file=sys.stderr)


def parse_es_csv(filename: str):
    """
    Parsers es.csv
    Format: "ID","Unknown","Index","Offset","Text"
    """
    print(f"Parsing {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        try:
            next(reader)  # Skip header row
        except StopIteration:
            return  # Empty file

        for i, row in enumerate(reader):
            try:
                if len(row) < 5:
                    continue

                id_col = row[0]
                index_col = row[2]
                text = clean_text(row[4])

                # Skip empty text or "Tribute Campaign" placeholders
                if not text or text == "Tribute Campaign":
                    continue

                dialog_list = [{"role": "npc", "text": text}]
                record = create_record(
                    id_suffix=f"{id_col}_{index_col}",
                    source_file=filename,
                    dialog_list=dialog_list,
                )
                if record:
                    yield record
            except Exception as e:
                print(f"Error parsing {filename} line {i+2}: {e}", file=sys.stderr)


def main():
    """
    Main function to parse all specified files and *append* to a .jsonl file.
    Resolves data paths relative to this script:
    humanized-npc-llm/humanized-npc-llm/data_engineering/data/raw
    """
    # Where is this script?
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Data root is: <script_dir>/data/raw
    data_root = os.path.join(script_dir, "data", "raw")

    # Put output next to the script (you can change this if you like)
    output_filename = os.path.join(script_dir, "npc_dialogue_records.jsonl")

    print(f"Data root resolved to: {data_root}")
    print(f"Output will be written to: {output_filename}")
    print(f"Script directory: {script_dir}")

    # Filenames we expect in data/raw
    files_to_process = [
        "battlespire.txt",
        "dagerfall.txt",
        "morrowwind.txt",
        "oblivion.txt",
        "arena_city_greetings.txt",
        "Arena_EQUIP.txt",
        "Arena_MUGUILD.txt",
        "arena_obj.txt",
        "arena_other_dialouges.txt",
        "Arena_SELLING.txt",
        "Arena_TAVERN.txt",
        "skyrim.txt",
        "es.csv",
    ]

    # Map *basenames* to parsers (simpler than full paths)
    parser_mapping = {
        "battlespire.txt": parse_battlespire,
        "dagerfall.txt": parse_dagerfall,
        "morrowwind.txt": parse_morrowwind,
        "oblivion.txt": parse_oblivion,
        "arena_city_greetings.txt": parse_arena_city_greetings,
        "Arena_EQUIP.txt": parse_arena_snippets,
        "Arena_MUGUILD.txt": parse_arena_snippets,
        "arena_obj.txt": parse_arena_obj,
        "arena_other_dialouges.txt": parse_arena_other_dialogues,
        "Arena_SELLING.txt": parse_arena_snippets,
        "Arena_TAVERN.txt": parse_arena_snippets,
        "skyrim.txt": parse_skyrim,
        "es.csv": parse_es_csv
    }

    # Build absolute paths
    abs_files_to_process = [os.path.join(data_root, name) for name in files_to_process]

    # Filter to only files that actually exist
    available_files = [f for f in abs_files_to_process if os.path.exists(f)]
    if not available_files:
        print(f"Error: None of the target files were found under {data_root}", file=sys.stderr)
        print("Please verify your data/raw directory structure.", file=sys.stderr)
        return

    print(f"Found {len(available_files)} files to process. Appending to {output_filename}...")

    total_records = 0

    with open(output_filename, "a", encoding="utf-8") as outfile:
        for filename in available_files:
            basename = os.path.basename(filename)
            parser_func = parser_mapping.get(basename)
            if not parser_func:
                print(f"Warning: No parser defined for {basename}. Skipping.", file=sys.stderr)
                continue

            try:
                file_records = 0
                for record in parser_func(filename):
                    outfile.write(json.dumps(record) + "\n")
                    total_records += 1
                    file_records += 1
                print(f"Finished parsing {basename}, found and appended {file_records} records.")
            except Exception as e:
                print(f"Critical error while parsing {basename}: {e}", file=sys.stderr)

    print("\n-------------------------------------------------")
    print(f"Done. Appended a total of {total_records} new records.")
    print(f"All records are in {output_filename}")
    print("-------------------------------------------------")

if __name__ == "__main__":
    main()
