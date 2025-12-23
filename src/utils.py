# Reward functions
import re
import torch
import random
import fast_regex
import Levenshtein
import string


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def extract_regex(text: str) -> str:
    matches = re.findall(r"```(?:regex)?\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
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


def is_valid_regex(expression: str) -> bool:
    metachars = set(r"[](){}*+?|^")
    return any(char in metachars for char in expression)


def soft_format_reward_func(prompts, completions, **kwargs):
    """Big negative reward for generating an invalid regex"""
    # responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for r in completions:
        regex = extract_regex(r)
        try:
            re.compile(regex)
            if not is_valid_regex(regex):
                score = -1.0
            else:
                score = 0.01  # a little reward for generating a valid extractable regex
        except:
            score = -1.0  # big negative reward for generating a regex that doesn't compile
        rewards.append(score)

    return rewards


def regex_similarity_reward_func(prompts, completions, **kwargs):
    """Measure the distance between generated regex and actual regex and give reward based on how close the generation is to the ground truth"""
    ground_truth_regexes = kwargs["reference_regex"]
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


def reasoning_length_penalty_func(prompts, completions, **kwargs):
    rewards = []
    for content in completions:

        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)

        if think_match:
            thinking_content = think_match.group(1)
            penalty = (len(thinking_content) / 2000) * 0.01
            rewards.append(-penalty) # We want the model to not think too much
        elif "<think>" in content and "</think>" not in content:
            rewards.append(-2.0) # the model was cut short while generating the response: not ideal
        else:
            rewards.append(0.0) # But still want it to think
    
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
        # final_score = (
        #     2 * (p_score * n_score) / (p_score + n_score + 1e-6)
        # )  # if p_score is 0, the whole thing becomes zero

        final_score = (p_score * 0.7) + (n_score * 0.3)

        # final_score += random.uniform(-0.001, 0.001)

        # Penalize the model for each new token it generates to force for a more concise thinking approach
        # token_count = len(regex) / 4  # rough token estimate
        # penalty = 0.0005 * token_count
        # final_score -= penalty

        rewards.append(final_score)

    return rewards


def name_generator(size: int = 12) -> str:
    return "".join(
        [random.choice(string.ascii_letters + string.digits) for _ in range(size)]
    )


class LoggingRewardCallback:
    def __init__(self, reward_funcs, log_every=10):
        self.reward_funcs = reward_funcs
        self.log_every = log_every
        self.call_count = 0
        self.__name__ = "logging_callback"

    def __call__(self, prompts, completions, **kwargs):
        if self.call_count % self.log_every == 0:
            try:
                # 1. Calculate and store all rewards
                rewards_data = {}
                total_rewards = torch.zeros(len(completions))

                for func in self.reward_funcs:
                    res = func(prompts=prompts, completions=completions, **kwargs)

                    # Standardize inputs (list -> tensor)
                    if isinstance(res, list):
                        res = torch.tensor(res)
                    elif isinstance(res, torch.Tensor):
                        res = res.cpu()

                    # Store specific scores and add to total
                    rewards_data[func.__name__] = res
                    total_rewards += res

                # 2. Find the index of the best generation based on TOTAL score
                best_idx = total_rewards.argmax().item()
                best_completion = completions[best_idx]

                # 3. Print the breakdown
                print("\n" + "=" * 60)
                print(f"STEP {self.call_count} | BEST GENERATION BREAKDOWN")
                print("-" * 60)

                # Print individual function scores for the winner
                for func_name, scores in rewards_data.items():
                    score = scores[best_idx].item()
                    print(f"{func_name:<35}: {score:.4f}")

                print("-" * 60)
                print(f"{'TOTAL REWARD':<35}: {total_rewards[best_idx].item():.4f}")
                print("=" * 60)
                print(f"OUTPUT:\n{best_completion}")
                print("=" * 60 + "\n")

            except Exception as e:
                print(f"Error in logging callback: {e}")

        self.call_count += 1
        # Return zeros so this callback doesn't affect training gradients
        return [0.0] * len(completions)
