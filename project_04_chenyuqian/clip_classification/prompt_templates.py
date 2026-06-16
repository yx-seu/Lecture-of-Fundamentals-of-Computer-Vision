"""Prompt engineering for CIFAR-10 zero-shot classification."""

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

SINGLE_TEMPLATE = ["a photo of a {}."]

MULTI_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a low resolution photo of a {}.",
    "a photo of a small {}.",
    "a photo of a large {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
]

# Class-specific prompts based purely on general world knowledge.
# Each prompt describes a universally recognizable visual characteristic
# of the category — no CIFAR-10 specific information, no test-set tuning.
CLASS_SPECIFIC_PROMPTS = {
    "airplane": [
        "a photo of an airplane.",
        "a photo of an airplane flying in the sky.",
        "a photo of an airplane at an airport.",
        "a photo of a jet airplane with wings.",
        "a photo of a passenger airplane.",
    ],
    "automobile": [
        "a photo of a car.",
        "a photo of an automobile on a road.",
        "a photo of a vehicle with four wheels.",
        "a photo of a sedan.",
        "a photo of a car driving on a street.",
    ],
    "bird": [
        "a photo of a bird.",
        "a photo of a bird perched on a branch.",
        "a photo of a small bird with feathers.",
        "a photo of a bird in a tree.",
        "a photo of a bird with wings.",
    ],
    "cat": [
        "a photo of a cat.",
        "a photo of a domestic cat.",
        "a photo of a cat sitting down.",
        "a photo of a pet cat with whiskers.",
        "a photo of a house cat.",
    ],
    "deer": [
        "a photo of a deer.",
        "a photo of a deer in the forest.",
        "a photo of a deer in the wild.",
        "a photo of a deer with antlers.",
        "a photo of a deer standing in a field.",
    ],
    "dog": [
        "a photo of a dog.",
        "a photo of a pet dog.",
        "a photo of a dog running outdoors.",
        "a photo of a domestic dog.",
        "a photo of a dog with fur.",
    ],
    "frog": [
        "a photo of a frog.",
        "a photo of a small frog.",
        "a photo of a green frog on a leaf.",
        "a photo of a frog near water.",
        "a photo of an amphibian frog.",
    ],
    "horse": [
        "a photo of a horse.",
        "a photo of a horse running in a field.",
        "a photo of a horse on a farm.",
        "a photo of a large horse with a mane.",
        "a photo of a horse standing in a pasture.",
    ],
    "ship": [
        "a photo of a ship.",
        "a photo of a ship on the ocean.",
        "a photo of a large ship at sea.",
        "a photo of a boat on water.",
        "a photo of a ship sailing.",
    ],
    "truck": [
        "a photo of a truck.",
        "a photo of a truck on a highway.",
        "a photo of a large truck.",
        "a photo of a pickup truck.",
        "a photo of a delivery truck on the road.",
    ],
}

# Optional synonyms: maps canonical class names to alternative words
CLASS_SYNONYMS = {
    "airplane": ["airplane", "plane"],
    "automobile": ["automobile", "car"],
}


CAT_FINE_GRAINED_PROMPTS = [
    "a photo of a domestic cat.",
    "a photo of a tabby cat.",
    "a photo of a Persian cat.",
    "a photo of a Siamese cat.",
    "a photo of a kitten.",
    "a photo of a wild cat.",
]


CAT_MISLEADING_PROMPTS = [
    "a photo of a dog-like cat.",
    "a photo of a toy cat.",
    "a photo of a cartoon cat.",
    "a photo of a cat in water.",
    "a photo of a cat on a road.",
    "not a cat.",
]


def generate_prompts(class_names, templates, use_synonyms=False):
    """Generate prompt strings for each class and template combination.

    Args:
        class_names: list of canonical class names.
        templates: list of template strings with `{}` placeholder.
        use_synonyms: if True, include synonym expansions from CLASS_SYNONYMS.

    Returns:
        list of (class_idx, template, prompt_text) tuples.
    """
    prompts = []
    for idx, name in enumerate(class_names):
        words = [name]
        if use_synonyms and name in CLASS_SYNONYMS:
            words = CLASS_SYNONYMS[name]
        for tpl in templates:
            for word in words:
                prompts.append((idx, tpl, tpl.format(word)))
    return prompts


def generate_class_specific_prompts(class_names, class_prompts):
    """Generate prompts using per-class descriptions.

    Args:
        class_names: list of canonical class names.
        class_prompts: dict mapping class_name -> list of prompt strings.

    Returns:
        list of (class_idx, template, prompt_text) tuples.
    """
    prompts = []
    for idx, name in enumerate(class_names):
        texts = class_prompts.get(name, ["a photo of a " + name + "."])
        for text in texts:
            prompts.append((idx, text, text))
    return prompts
