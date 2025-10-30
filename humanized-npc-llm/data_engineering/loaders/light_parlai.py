def load_light_say_only(max_pairs=None):
    try:
        from parlai.core.params import ParlaiParser
        from parlai.core.worlds import create_task
    except Exception as e:
        raise RuntimeError("ParlAI not installed; pip install parlai") from e

    pp = ParlaiParser(True, True)
    pp.set_defaults(task="light_dialog", datatype="train")
    opt = pp.parse_args([])
    world = create_task(opt, None)

    count = 0
    world.reset()
    while True:
        if world.epoch_done():
            break
        world.parley()
        acts = world.get_acts() or []
        if not acts:
            continue
        msg = acts[0]
        text = (msg.get("text") or "").strip()
        labels = msg.get("labels") or msg.get("eval_labels") or []
        if not (text and labels):
            continue

        dialog = [{"role": "player", "text": text}, {"role": "npc", "text": labels[0]}]
        yield {
            "id": f"light_{count}",
            "source": "light",
            "split": "train",
            "persona": [],
            "world_facts": [],
            "context": {
                "npc_role": msg.get("character") or "generic",
                "location": msg.get("setting") or None,
            },
            "intent": "generic",
            "control": {"style": ["fantasy"]},
            "dialog": dialog,
            "meta": {
                "license": "mit",
                "dataset_ref": "https://parl.ai/docs/tasks.html#light-dialogue",
            },
        }
        count += 1
        if max_pairs and count >= max_pairs:
            break
