"""
Per-semantic-class captioning prompts for re-captioning the IAB **real** images
with a vision-language model (Qwen3.5-9B via Ollama).

Rationale
---------
The original ImageAttributionBench captions (Qwen-VL-Chat) are single, generic
sentences of ~20-50 words. Synthetic images generated from such thin prompts are
easy to tell apart from real ones, which makes the attribution task trivial.

To make the task harder we re-caption every real image with a *dense* prompt
(40-80 words, diffusion-prompt style) that could be used to regenerate a
visually similar image with FLUX / Stable Diffusion. Each semantic class gets a
prompt tailored to what matters for that domain (faces → facial attributes,
scenes → architecture/layout, objects → the dominant subject, ...).

Editing
-------
These strings are meant to be iterated on. Tune the attribute checklists or the
word budget per class; `caption_real_images.py` only needs `PROMPT_BY_SEMANTIC`.
"""

# ── Shared output contract ────────────────────────────────────────────────────
# Appended to every prompt so the model returns ONLY the caption, in a uniform
# diffusion-prompt style. Kept separate so the rules stay consistent across
# classes and are edited in one place.
_OUTPUT_RULES = (
    "OUTPUT INSTRUCTIONS — CRITICAL:\n"
    "- Respond with ONLY the final caption. No preamble, no explanation, no labels, no bullet points.\n"
    "- Do not write things like 'Here is the caption:' or 'Caption:' — start the caption immediately.\n"
    "- Do not think out loud or include any reasoning; output the caption directly.\n"
    "- Format: a single dense paragraph of 40-80 words, written as comma-separated descriptive "
    "phrases (diffusion model prompt style, not full sentences).\n"
    "- Your entire response must be the caption and nothing else."
)


def _prompt(role_intro: str, checklist: str, extra_rules: str = "") -> str:
    """Compose a class prompt: intro + attribute checklist + output contract."""
    body = f"{role_intro}\n\n{checklist}\n\n"
    if extra_rules:
        body += extra_rules + "\n\n"
    return body + _OUTPUT_RULES


# ── Faces (FFHQ, celebahq) ────────────────────────────────────────────────────
# Provided verbatim by the user — kept as-is (it already carries its own output
# rules, so we do NOT append _OUTPUT_RULES here).
FACIAL_CAPTION_PROMPT = (
    "You are a precise visual captioning assistant. Your goal is to produce a dense, "
    "faithful description of the face in this real photograph that could be used to "
    "regenerate a visually identical image with a text-to-image model like FLUX or Stable Diffusion.\n\n"

    "Analyze the following facial attributes:\n"
    "1. **Overall face shape**: (e.g., oval, round, square, heart-shaped, angular)\n"
    "2. **Skin**: tone (e.g., fair, olive, dark brown), texture (e.g., smooth, freckled, weathered), "
    "and any notable features (e.g., wrinkles, stubble, acne, moles)\n"
    "3. **Eyes**: shape (e.g., almond, round, hooded), size, color, lash density, "
    "and any distinctive traits (e.g., deep-set, heavy-lidded, light eyebrows)\n"
    "4. **Nose**: shape (e.g., straight, broad, upturned, aquiline), size relative to face\n"
    "5. **Mouth and lips**: lip fullness, shape of cupid's bow, expression (e.g., neutral, slight smile)\n"
    "6. **Facial expression**: the dominant emotion or mood conveyed\n"
    "7. **Hair** (if visible): color, texture, style, and how it frames the face\n"
    "8. **Lighting on face**: direction and quality (e.g., soft front light, harsh side light, golden hour)\n\n"

    "OUTPUT INSTRUCTIONS — CRITICAL:\n"
    "- Respond with ONLY the final caption. No preamble, no explanation, no labels, no bullet points.\n"
    "- Do not write things like 'Here is the caption:' or 'Caption:' — start the caption immediately.\n"
    "- Format: a single dense paragraph of 40-80 words, written as comma-separated descriptive phrases "
    "(diffusion model prompt style, not full sentences).\n"
    "- Do not mention the background, clothing, or anything outside the face and immediately framing hair.\n"
    "- Your entire response must be the caption and nothing else."
)


# ── Animals (AnimalFace: cat, dog, wild) ──────────────────────────────────────
ANIMAL_CAPTION_PROMPT = _prompt(
    "You are a precise visual captioning assistant. Your goal is to produce a dense, "
    "faithful description of the animal in this real photograph that could be used to "
    "regenerate a visually identical image with a text-to-image model like FLUX or Stable Diffusion.",

    "Analyze the following attributes:\n"
    "1. **Species / breed**: the animal and, if recognizable, its breed or type\n"
    "2. **Fur or coat**: dominant colors, pattern (e.g., tabby, spotted, solid), and texture "
    "(e.g., short, fluffy, wiry, glossy)\n"
    "3. **Head and face**: shape, ear form and position, snout/muzzle, and any markings\n"
    "4. **Eyes**: color, shape, and gaze direction\n"
    "5. **Distinctive features**: whiskers, mane, scars, collar, age cues (young/adult)\n"
    "6. **Pose and framing**: head orientation, body posture, how close the crop is\n"
    "7. **Lighting**: direction and quality on the animal\n"
    "8. **Immediate setting**: only a brief note on the surface or backdrop directly around the animal",
)


# ── Indoor / architectural scenes (LSUN: bedroom, church, classroom) ──────────
SCENE_CAPTION_PROMPT = _prompt(
    "You are a precise visual captioning assistant. Your goal is to produce a dense, "
    "faithful description of the scene in this real photograph that could be used to "
    "regenerate a visually similar image with a text-to-image model like FLUX or Stable Diffusion.",

    "Analyze the following attributes:\n"
    "1. **Type of space**: what kind of room or interior/exterior it is\n"
    "2. **Layout and perspective**: camera viewpoint, depth, vanishing lines, what is in foreground vs background\n"
    "3. **Architecture**: walls, ceiling, windows, columns, arches, floor, and their materials\n"
    "4. **Furniture and objects**: the main pieces present, their arrangement, style and materials\n"
    "5. **Color palette**: dominant and accent colors of the space\n"
    "6. **Lighting**: sources (windows, lamps), direction, warmth, and overall brightness/mood\n"
    "7. **Style and condition**: period/aesthetic (e.g., modern, ornate, worn, minimalist) and tidiness",
)


# ── Everyday complex scenes (COCO) ────────────────────────────────────────────
COCO_CAPTION_PROMPT = _prompt(
    "You are a precise visual captioning assistant. Your goal is to produce a dense, "
    "faithful description of this real everyday photograph that could be used to "
    "regenerate a visually similar image with a text-to-image model like FLUX or Stable Diffusion.",

    "Analyze the following attributes:\n"
    "1. **Main subjects**: the people, animals, and/or objects that dominate the scene, with counts\n"
    "2. **Actions and interactions**: what the subjects are doing\n"
    "3. **Attributes**: colors, clothing, materials, sizes, and notable details of each main subject\n"
    "4. **Spatial layout**: where subjects are relative to each other and to the frame\n"
    "5. **Setting**: the environment or location and relevant background elements\n"
    "6. **Lighting and time of day**: direction, quality, indoor/outdoor, weather if visible\n"
    "7. **Photographic style**: viewpoint, framing, depth of field, and overall mood",
)


# ── Single-object / ImageNet images ───────────────────────────────────────────
OBJECT_CAPTION_PROMPT = _prompt(
    "You are a precise visual captioning assistant. Your goal is to produce a dense, "
    "faithful description of the main subject in this real photograph that could be used to "
    "regenerate a visually similar image with a text-to-image model like FLUX or Stable Diffusion.",

    "Analyze the following attributes:\n"
    "1. **Dominant subject**: the single object, animal, or thing the photo is centered on\n"
    "2. **Category and identity**: what it is, as specifically as you can tell\n"
    "3. **Form**: shape, size, proportions, and orientation/pose\n"
    "4. **Surface**: colors, material, texture, patterns, and condition (new, worn, etc.)\n"
    "5. **Distinctive details**: parts, text/logos, or features that identify it\n"
    "6. **Immediate context**: the surface it rests on or the setting directly around it\n"
    "7. **Lighting and framing**: light direction/quality, viewpoint, depth of field, and mood",
)


# ── Semantic class → prompt ───────────────────────────────────────────────────
PROMPT_BY_SEMANTIC: dict[str, str] = {
    "FFHQ":        FACIAL_CAPTION_PROMPT,
    "celebahq":    FACIAL_CAPTION_PROMPT,
    "cat":         ANIMAL_CAPTION_PROMPT,
    "dog":         ANIMAL_CAPTION_PROMPT,
    "wild":        ANIMAL_CAPTION_PROMPT,
    "bedroom":     SCENE_CAPTION_PROMPT,
    "church":      SCENE_CAPTION_PROMPT,
    "classroom":   SCENE_CAPTION_PROMPT,
    "COCO":        COCO_CAPTION_PROMPT,
    "ImageNet-1k": OBJECT_CAPTION_PROMPT,
}
