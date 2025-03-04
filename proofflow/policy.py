from torch import nn
from typing import Optional, List
from transformers import PreTrainedTokenizer
import torch

MAX_OUTPUT_LEN = 500

class Policy:
    """
    A policy generating subsequent tactics given a goal and optionally the tactics so far

    Will predict tactics querying the model using the prompt
    GOAL <goal> PROOFSTEP

    Optionally, if tactics so far are given, will instead prompt as
    GOAL <goal> TACTICS <tactics> PROOFSTEP
    """

    def __init__(self, model: nn.Module, eos_id: int, proof_step_id: int, goal_id: int, tactics_id: int,
                 tactics_sep_id: int, tokenizer: PreTrainedTokenizer):
        """

        :param model: The underlying model to use
        :param eos_id: The eos token id, marking end of generation
        :param proof_step_id: The proof step token id
        :param goal_id: The goal token id
        :param tactics_id: The tactics token id
        :param tactics_sep_id: The tactics separation token id
        :param tokenizer: The tokenizer to use
        """
        self.model = model
        self.eos_token = eos_id
        self.proof_step_id = proof_step_id
        self.goal_id = goal_id
        self.tactics_id = tactics_id
        self.tactics_sep_id = tactics_sep_id
        self.tokenizer = tokenizer
        self.softmax = nn.Softmax()

    def next_tactic(self, goal: str, tactics_so_far: Optional[List[str]] = None, temperature: float = 0.0) -> str:
        """Predict the subsequent tactic for the given goal.

        :param goal: The goal to predict the tactic for.
        :param tactics_so_far: Optional list of tactics so far
        :param temperature: The temperature to use, 0 for greedy sampling
        :return: The subsequent tactic
        """
        prompt = self._build_prompt(goal, tactics_so_far)
        prompt_tensor = torch.tensor(prompt)
        token = None
        tactic = []
        idx = 0
        while token != self.eos_token and idx < MAX_OUTPUT_LEN:
            logits = self.model(prompt_tensor)[-1] # we only use the final one, the rest is previous tokens only used in training
            if temperature > 0.0:
                softmaxed = self.softmax(logits / temperature)
                token = softmaxed.sample()
            else:
                token = logits.argmax(dim=0)
            prompt_tensor = torch.cat([prompt_tensor, token[None]])
            tactic.append(token.item())
            idx += 1
        tactic.pop(-1)
        return self.tokenizer.decode(tactic)

    def _build_prompt(self, goal: str, tactics_so_far: Optional[List[str]] = None) -> List[int]:
        goal_ids: List[int] = self.tokenizer.encode(goal)
        if tactics_so_far is not None:
            tactics: List[List[int]] = [self.tokenizer.encode(tactic) for tactic in tactics_so_far]
            for idx, tactic in enumerate(tactics):
                if idx < len(tactics) - 1:
                    tactic.append(self.tactics_sep_id)
            tactics_flat: List[int] = sum(tactics, [])
            return [self.goal_id] + goal_ids + [self.tactics_id] + tactics_flat + [
                self.proof_step_id]
        return [self.goal_id] + goal_ids + [self.proof_step_id]
