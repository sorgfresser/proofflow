from torch import nn
from typing import Optional, List
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from proofflow.data import TrainingSample

MAX_OUTPUT_LEN = 500


class Policy:
    """
    A policy generating subsequent tactics given a proof state and optionally the tactics so far

    Will predict tactics querying the model using the prompt
    PROOFSTATE <state> PROOFSTEP

    Optionally, if tactics so far are given, will instead prompt as
    PROOFSTATE <state> TACTICS <tactics> PROOFSTEP
    """

    def __init__(self, model: nn.Module, eos_id: int, proof_step_id: int, proof_state_id: int, tactics_id: int,
                 tactics_sep_id: int, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
        """

        :param model: The underlying model to use
        :param eos_id: The eos token id, marking end of generation
        :param proof_step_id: The proof step token id
        :param proof_state_id: The proof state token id
        :param tactics_id: The tactics token id
        :param tactics_sep_id: The tactics separation token id
        :param tokenizer: The tokenizer to use
        """
        self.model = model
        self.eos_token = eos_id
        self.proof_step_id = proof_step_id
        self.proof_state_id = proof_state_id
        self.tactics_id = tactics_id
        self.tactics_sep_id = tactics_sep_id
        self.tokenizer = tokenizer
        self.softmax = nn.Softmax()
        self.loss_fn = nn.CrossEntropyLoss()

    def next_tactic(self, proof_state: str, tactics_so_far: Optional[List[str]] = None,
                    temperature: float = 0.0) -> str:
        """Predict the subsequent tactic for the given proof state (which might have multiple goals)

        :param proof_state: The proof state used to predict the tactic for.
        :param tactics_so_far: Optional list of tactics so far
        :param temperature: The temperature to use, 0 for greedy sampling
        :return: The subsequent tactic
        """
        prompt = self._build_prompt(proof_state, tactics_so_far)
        prompt_tensor = torch.tensor(prompt)
        token = None
        tactic = []
        idx = 0
        while token != self.eos_token and idx < MAX_OUTPUT_LEN:
            logits = self.model(prompt_tensor)[
                -1]  # we only use the final one, the rest is previous tokens only used in training
            if temperature > 0.0:
                softmaxed = self.softmax(logits / temperature)
                token = torch.multinomial(softmaxed, 1).squeeze()
            else:
                token = logits.argmax(dim=0)
            prompt_tensor = torch.cat([prompt_tensor, token[None]])
            tactic.append(token.item())
            idx += 1
        tactic.pop(-1)
        return self.tokenizer.decode(tactic)

    def _build_prompt(self, proof_state: str, tactics_so_far: Optional[List[str]] = None) -> List[int]:
        state_ids: List[int] = self.tokenizer.encode(proof_state)
        if tactics_so_far is not None:
            tactics: List[List[int]] = [self.tokenizer.encode(tactic) for tactic in tactics_so_far]
            for idx, tactic in enumerate(tactics):
                if idx < len(tactics) - 1:
                    tactic.append(self.tactics_sep_id)
            tactics_flat: List[int] = sum(tactics, [])
            return [self.proof_state_id] + state_ids + [self.tactics_id] + tactics_flat + [
                self.proof_step_id]
        return [self.proof_state_id] + state_ids + [self.proof_step_id]

    def train_batch(self, batch: list[TrainingSample], loss_on_prompt: bool = False):
        """Train on one single batch of training samples.

        :param batch:
        :param loss_on_prompt: Whether to also compute language modelling loss on prompt tokens.
        :return:
        """
        prompts = [self._build_prompt(sample.proof_state, sample.tactics_so_far) for sample in batch]
        tactics = [self.tokenizer.encode(sample.tactic) for sample in batch]

        full = {
            "input_ids": [prompt + tactic + [self.eos_token] for prompt, tactic in zip(prompts, tactics, strict=True)]}

        padded = self.tokenizer.pad(full, padding_side="right", return_attention_mask=True, return_tensors="pt")
        input_ids = padded.input_ids
        attention_mask = padded.attention_mask[:, 1:]
        input_ids = input_ids.to(self.model.device)
        logits = self.model(input_ids)[:, :-1, :]
        labels = input_ids[:, 1:]
        labels = labels.masked_fill(~attention_mask.bool(), -100)
        # Only compute loss for the part after the prompt
        if not loss_on_prompt:
            for i in range(len(prompts)):
                labels[i, :len(prompts[i]) - 1] = -100
        return self.loss_fn(logits.transpose(2, 1), labels)
