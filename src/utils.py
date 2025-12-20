# Reward functions
import re
import random
import fast_regex


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def extract_regex(text: str) -> str:
    match = re.search(r"```(?:regex)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


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
            rewards.append(-1.0 + random.uniform(-0.005, 0.005))
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
