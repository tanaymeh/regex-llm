# Reward functions
import re
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
            rewards.append(0.1)  # small reward for generating a valid regex
        except:
            rewards.append(-1.0)  # big negative reward for generating an invalid regex
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
            rewards.append(-1.0)
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

        rewards.append((p_score + n_score) / 2.0)

    return rewards
