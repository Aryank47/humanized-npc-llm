# Humanized NPC-LLM: Bridging Game NPCs and Human Conversation

This project fine-tunes a 3-4B parameter instruction-tuned language model 
to generate short, persona-consistent dialogue for video game Non-Player 
Characters (NPCs) that feel human rather than robotic. Using Parameter-
Efficient Fine-Tuning (PEFT/QLoRA), we blend conversational naturalness 
from PersonaChat with structural NPC patterns from game dialogue datasets 
(Skyrim, LIGHT) to create NPCs that can handle quests, trades, and 
contextual interactions while maintaining personality and emotional depth.

Key Features:
- Humanized NPCs: Blend of natural conversation (40%) + NPC structure (25%) + hybrid roleplay (35%)
- Efficient training: QLoRA on consumer GPUs (RTX 4060)
- Structured output: JSON schema with persona, dialogue, intent, and world grounding
- Production-ready: <80 token responses, <200ms latency
- Comprehensive evaluation: Persona consistency, hallucination control, NPC authenticity

Datasets: Synthetic-Persona-Chat, LIGHT and all data from https://www.imperial-library.info/out-of-game/game-data