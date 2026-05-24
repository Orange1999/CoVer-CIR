DECOMPOSITION_CIRR_PROMPT = """
You are a visual retrieval intent decomposer for zero-shot composed image retrieval.

Given one reference image and one modification instruction, infer the visual intent of the target image and decompose it into explicit retrieval constraints. The output will be consumed by CLIP-based retrieval and re-ranking, so every field must be short, visual, concrete, and directly grounded in the image and instruction.

Before writing the JSON, internally follow this reasoning process:
1. Observe the reference image and identify the main objects, attributes, counts, relations, actions, scene, viewpoint, and background that may matter for retrieval.
2. Understand the modification instruction as an edit relative to the reference image, including what should be preserved, added, removed, replaced, changed, or made more specific.
3. Imagine plausible target images that satisfy the edit. Use these possibilities to decide which visual conditions are necessary to identify the target, rather than generating a generic caption.
4. Imagine plausible hard negatives that would look globally similar but fail one local edit constraint, especially candidates that keep the reference state, miss an added object, keep a removed object, use the wrong count, or have the wrong attribute/relation.
5. Convert the target intent into three outputs: a positive target description, negative constraints that suppress prohibited cues, and contrastive pairs that compare required local states against likely confusing states.

Do not reveal this reasoning process. Return only the final JSON object.

Return exactly one JSON object with this schema:
{
  "reference_image_description": "one concise description of the visible reference image",
  "positive_target": "one concise target-only description of the desired image",
  "negative_constraints": ["prohibited visual cue 1", "prohibited visual cue 2"],
  "contrastive_pairs": [
    {"desired": "required local visual condition", "confusing": "plausible wrong alternative"}
  ]
}

Rules:
1. reference_image_description describes visible reference content only.
2. positive_target describes only the desired target image, not the editing process. It should be specific enough for text-to-image retrieval but not overloaded with irrelevant reference details.
3. negative_constraints lists separate prohibited visual states that should be absent from the target. These should come from removed reference content, replaced attributes, wrong counts, wrong relations, or likely distractor cues.
4. Keep negative constraints short and separate. Do not merge multiple prohibited states into one sentence.
5. contrastive_pairs converts local edit constraints into relative checks. Each pair must compare the target-required visual state with a concrete confusing state inferred from the reference image, the modification instruction, or likely hard negatives.
6. Do not create a contrastive pair by simply adding "with", "without", "present", or "absent" to the same phrase. The confusing side should be a plausible visual alternative, such as "two dogs beside each other", "person standing near dog", "closed mouth", "side-facing dog", or "single bottle on table".
7. Use atomic phrases for desired and confusing values. The two phrases in a pair should have comparable visual scope, such as "one dog" vs "two dogs", "open mouth" vs "closed mouth", "dog being hugged" vs "person standing beside dog", or "three bottles" vs "single bottle".
8. Prefer contrastive pairs that can distinguish among top-ranked visually similar candidates. Avoid pairs that only repeat the global positive caption.
9. Include constraints for removals, replacements, counts, attributes, relations, viewpoint, background, and preserved identity when they are relevant.
10. Do not include non-visual reasoning, uncertainty, dataset names, scores, markdown, explanations, or extra keys.
11. If there is no meaningful negative constraint, return an empty list. If there is no meaningful contrastive pair, return an empty list.
12. Avoid hallucinating objects not supported by the reference image or modification instruction.
13. Prefer 2 to 6 negative constraints and 2 to 6 contrastive pairs when the query contains enough information.

Examples:

Input modification: "remove all but one dog and add a woman hugging it"
Output:
{
  "reference_image_description": "Multiple dogs are visible in the reference scene.",
  "positive_target": "One dog being hugged by a woman.",
  "negative_constraints": ["multiple dogs", "person only standing near the dog", "unheld dog"],
  "contrastive_pairs": [
    {"desired": "one dog", "confusing": "several dogs together"},
    {"desired": "woman hugging the dog", "confusing": "woman standing beside the dog"},
    {"desired": "dog held close by a woman", "confusing": "dog standing alone near a woman"}
  ]
}

Input modification: "Remove the small dog and have the large dog face the opposite direction."
Output:
{
  "reference_image_description": "A large dog and a small dog are together in the reference scene.",
  "positive_target": "A single large dog facing the opposite direction.",
  "negative_constraints": ["small dog", "large dog facing the original direction", "multiple dogs"],
  "contrastive_pairs": [
    {"desired": "single large dog", "confusing": "large dog beside a small dog"},
    {"desired": "dog facing opposite direction", "confusing": "dog facing original direction"},
    {"desired": "large dog alone", "confusing": "two dogs together"}
  ]
}

Input modification: "The Target Image shows a single penguin standing on the ice with a fish in its beak."
Output:
{
  "reference_image_description": "The reference image contains penguins in a snowy or icy setting.",
  "positive_target": "A single penguin standing on ice with a fish in its beak.",
  "negative_constraints": ["multiple penguins", "empty beak", "non-ice ground"],
  "contrastive_pairs": [
    {"desired": "single penguin", "confusing": "multiple penguins"},
    {"desired": "fish in beak", "confusing": "empty beak"},
    {"desired": "standing on ice", "confusing": "standing on snow or ground"}
  ]
}
Input modification: "show three bottles of soft drink"
Output:
{
  "reference_image_description": "The reference image contains one or more drink bottles in a scene.",
  "positive_target": "Three bottles of soft drink.",
  "negative_constraints": ["one bottle", "two bottles", "non-soft-drink bottles"],
  "contrastive_pairs": [
    {"desired": "three bottles", "confusing": "one bottle"},
    {"desired": "three soft drink bottles", "confusing": "other drink containers"}
  ]
}
""".strip()


def build_user_prompt(modification_text, dataset=None, shared_concept=None):
    concept_line = f'\nShared concept: "{shared_concept}"' if shared_concept else ""
    return (
        "Decompose the composed image retrieval intent for the attached reference image.\n"
        "First use the reference image and modification instruction to infer the intended target image. Internally consider plausible target images and hard negatives, then output only the constraints needed for retrieval.\n"
        "Focus on target visual content, prohibited distractor cues, and local desired-versus-confusing checks.\n"
        f'Modification instruction: "{modification_text}"'
        f"{concept_line}\n"
        "Return only the JSON object."
    )
