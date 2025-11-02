import logging
import re

logger = logging.getLogger(__name__)
TAG_RE = re.compile(r'_(self|partner)_(say|act|emote)\s*', flags=re.I)

def _clean_light_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # remove control tags like _self_say, _partner_act, etc.
    s = TAG_RE.sub("", s)
    # unify whitespace and stray underscores
    s = s.replace("_comma_", ",").replace("_period_", ".").replace("_exclamation_", "!").replace("_question_", "?")
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_light_say_only(max_pairs=None):
    """
    Robust LIGHT loader that avoids create_task() signature differences.
    Iterates the DefaultTeacher directly and yields (player, npc) pairs.
    """
    try:
        from parlai.core.params import ParlaiParser
        from parlai.tasks.light_dialog.agents import DefaultTeacher
    except Exception as e:
        raise RuntimeError("ParlAI not installed; pip install parlai") from e

    # Build ParlAI options: train split, batchsize=1 for sequential iteration
    pp = ParlaiParser(add_parlai_args=True, add_model_args=False)
    pp.set_defaults(task="light_dialog", datatype="train", batchsize=1, num_threads=1)
    opt = pp.parse_args([])

    # Teacher-based iteration (no World, no user_agents needed)
    try:
        teacher = DefaultTeacher(opt)
    except Exception as e:
        logger.error(f"LIGHT DefaultTeacher init failed: {e}")
        return

    if hasattr(teacher, "reset"):
        try:
            teacher.reset()
        except Exception:
            pass

    count = 0
    while True:
        # Stop conditions
        if hasattr(teacher, "epoch_done") and teacher.epoch_done():
            break
        if max_pairs and count >= max_pairs:
            break

        try:
            ex = teacher.act()
        except StopIteration:
            break
        except Exception as e:
            logger.warning(f"LIGHT teacher.act() error at {count}: {e}")
            break

        text = _clean_light_text(ex.get("text") or "")
        labels = ex.get("labels") or ex.get("eval_labels") or []
        if not (text and labels and isinstance(labels[0], str) and labels[0].strip()):
            continue

        reply = _clean_light_text(labels[0])
        if not reply:
            continue

        role = ex.get("character") or "generic"
        location = ex.get("setting") or None

        dialog = [
            {"role": "player", "text": text},
            {"role": "npc", "text": reply},
        ]
        
        # build context only with non-empty strings
        ctx = {}
        if isinstance(role, str) and role.strip():
            ctx["npc_role"] = role.strip()
        if isinstance(location, str) and location.strip():
            ctx["location"] = location.strip()

        yield {
            "id": f"light_{count}",
            "source": "light",
            "split": "train",
            "persona": [],
            "world_facts": [],
            "context": ctx,
            "intent": "generic",
            "control": {"style": ["fantasy"]},
            "dialog": dialog,
            "meta": {
                "license": "mit",
                "dataset_ref": "https://parl.ai/docs/tasks.html#light-dialogue",
            },
        }

        count += 1
