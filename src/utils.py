# Reward functions
import re
import random
import fast_regex
import Levenshtein
import string


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def extract_regex(text: str) -> str:
    match = re.search(r"```(?:regex)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_boxed(text):
    """Extract content from \boxed{} including nested braces"""
    pattern = r"\\boxed\{"
    matches = []

    for match in re.finditer(pattern, text):
        start = match.end()
        brace_count = 1
        i = start

        while i < len(text) and brace_count > 0:
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
            i += 1

        if brace_count == 0:
            matches.append(text[start : i - 1])

    return matches


def soft_format_reward_func(prompts, completions, **kwargs):
    """Big negative reward for generating an invalid regex"""
    # responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for r in completions:
        regex = extract_regex(r)
        try:
            re.compile(regex)
            score = 0.5  # small reward for generating a valid regex
        except:
            if "```" in r:
                score = -0.5  # partial credit for correct structure
            else:
                score = -1.0  # big negative reward for generating an invalid regex
        score += random.uniform(
            -0.005, 0.005
        )  # small noise to ensure we don't get 0 std
        rewards.append(score)

    return rewards


def regex_similarity_reward_func(prompts, completions, **kwargs):
    """Measure the distance between generated regex and actual regex and give reward based on how close the generation is to the ground truth"""
    ground_truth_regexes = [x["reference_regex"] for x in kwargs["data"]]
    rewards = []

    for content, gt_regex in zip(completions, ground_truth_regexes):
        pred_regex = extract_regex(content)

        distance = Levenshtein.distance(pred_regex, gt_regex)
        max_len = max(len(pred_regex), len(gt_regex))

        if max_len == 0:
            score = 0.0
        else:
            score = 1.0 - (distance / max_len)  # also the 'similarity'

        rewards.append(score)

    return rewards


def correctness_reward_func(prompts, completions, **kwargs):
    positive_cases_batch = kwargs["positive_test_cases"]
    negative_cases_batch = kwargs["negative_test_cases"]

    rewards = []

    for content, p_cases, n_cases in zip(
        completions, positive_cases_batch, negative_cases_batch
    ):
        regex = extract_regex(content)

        try:
            re.compile(regex)
        except:
            rewards.append(0.0)
            continue

        match_limit = 50000  # Stricter limit for speed

        # Calculate Recall (Positive Cases)
        p_matches = 0
        for t in p_cases:
            if fast_regex.fullmatch(regex, t, match_limit):
                p_matches += 1

        # Calculate Precision (Negative Cases)
        n_non_matches = 0
        for t in n_cases:
            if not fast_regex.fullmatch(regex, t, match_limit):
                n_non_matches += 1

        p_score = p_matches / len(p_cases) if p_cases else 0
        n_score = n_non_matches / len(n_cases) if n_cases else 0
        final_score = (
            2 * (p_score * n_score) / (p_score + n_score + 1e-6)
        )  # if p_score is 0, the whole thing becomes zero

        rewards.append(final_score + random.uniform(-0.001, 0.001))

    return rewards


def name_generator(size: int = 12) -> str:
    return "".join(
        [random.choice(string.ascii_letters + string.digits) for _ in range(size)]
    )
