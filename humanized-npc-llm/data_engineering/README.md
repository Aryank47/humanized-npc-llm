# Data engineering Setup

## File extraction

1. Fine Tuning data setup
    - Go to data/processed/v2.zip
    - unzip it and add it the v2 folder in data/processed/
    - the final directory structure will look like data/processed/v2/train and test and val .jsonl

2. Raw data setup and parsing
    - Go to data/raw/raw.zip
    - unzip it and add it add all the individual files 13 in total to data/raw/
    - **NOTE: there is one file missing i.e es.csv from the raw.zip as it was too large, so you can download the csv file from here https://www.imperial-library.info/out-of-game/game-data under the **Elder Scrolls Online** header and add it to the data/raw/ as es.csv**
    - then
        ```bash
        cd humanized-npc-llm/humanized-npc-llm/data_engineering
        python parse_game_data.py
        ```
    - This will create npc_dialogue_records.jsonl in data_engineering/ folder which will be the combination of all files from raw/ parsed as per the schema/npc_schema.json
    

