# Humanized NPC-LLM: Fine-Tuning a Small LM for Persona-Consistent NPC (Non-Player Character) Dialogue

Humanised NPC-LLM project fine-tunes a small (3–4B) instruction-tuned language model to produce persona-consistent, humanised NPC dialogue for games using the PEFT approach, Quantised Low Rank Adaptation Technique (QLoRA). It uses game dialogue corpora from “The Imperial Library” (Skyrim transcripts, Elder Scrolls, Daggerfall, etc) and outputs structured JSON responses (persona + dialogue + intent + world facts). The general idea is “can we move beyond rigid, pre-scripted NPC dialogue and instead enable NPCs to generate contextually rich, adaptive responses that better simulate human conversational behaviour?”

Key Features:
- Humanized NPCs: Blend of natural conversation (40%) + NPC structure (25%) + hybrid roleplay (35%)
- Efficient training: QLoRA
- Structured output: JSON schema with persona, dialogue, intent, and world grounding
- Production-ready: <80 token responses, <200ms latency
- Comprehensive evaluation: Persona consistency, hallucination control, NPC authenticity

Datasets: Synthetic-Persona-Chat, LIGHT and all data from https://www.imperial-library.info/out-of-game/game-data