from torch import nn
from typing import Optional, List, Union
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding
import torch
from proofflow.data import TrainingSample
from pathlib import Path
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm import MambaLMHeadModel
from dataclasses import asdict

MAX_OUTPUT_LEN = 500


class MambaLMHeadModelWrapper(MambaLMHeadModel):

    def __init__(self, config, *args, device=None, dtype=None, is_gflownet: bool = False, **kwargs):
        super().__init__(config, *args, device=device, dtype=dtype, **kwargs)
        if is_gflownet:
            self.back_head = nn.Linear(config.d_model, config.vocab_size, bias=False, device=device, dtype=dtype)
            self.z_head = nn.Linear(config.d_model, 1, bias=True, device=device, dtype=dtype)
        else:
            self.back_head, self.z_head = None, None

    def forward(self, x):
        return super().forward(x).logits

    def p_b(self, input_ids):
        hidden_states = self.backbone(input_ids)
        back_logits = self.back_head(hidden_states)
        return back_logits

    def log_z(self, input_ids):
        hidden_states = self.backbone(input_ids)[:, -1]
        lm_logits = self.z_head(hidden_states)
        return lm_logits


class Policy:
    """
    A policy generating subsequent tactics given a proof state and optionally the tactics so far

    Will predict tactics querying the model using the prompt
    PROOFSTATE <state> PROOFSTEP

    Optionally, if tactics so far are given, will instead prompt as
    PROOFSTATE <state> TACTICS <tactics> PROOFSTEP
    """

    def __init__(self, model: nn.Module, eos_id: int, proof_step_id: int, proof_state_id: int, tactics_id: int,
                 tactics_sep_id: int, proofstate_sep_id: int, goals_sep_id: int, successful_proof_token: int,
                 incomplete_proof_token: int, invalid_proof_token: int,
                 tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                 device: str = "cpu"):
        """

        :param model: The underlying model to use
        :param eos_id: The eos token id, marking end of generation
        :param proof_step_id: The proof step token id
        :param proof_state_id: The proof state token id
        :param tactics_id: The tactics token id
        :param tactics_sep_id: The tactics separation token id
        :param tokenizer: The tokenizer to use
        :param device: The device the model is on
        """
        self.model = model
        self.eos_token = eos_id
        self.proof_step_id = proof_step_id
        self.proof_state_id = proof_state_id
        self.tactics_id = tactics_id
        self.tactics_sep_id = tactics_sep_id
        self.proofstate_sep_id = proofstate_sep_id
        self.goals_sep_id = goals_sep_id
        self.successful_proof_token = successful_proof_token
        self.incomplete_proof_token = incomplete_proof_token
        self.invalid_proof_token = invalid_proof_token
        self.tokenizer = tokenizer
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device

    def next_tactic(self, proof_states: Union[List[str], str], tactics_so_far: Optional[Union[List[List[str]], List[str]]] = None,
                          previous_proof_states: Optional[Union[List[List[str]], List[str]]] = None,
                          temperature: float = 0.0, max_new_tokens: int = 20) -> Union[List[str], str]:
        """Predict the subsequent tactics for the given proof states (which might have multiple goals)

        :param proof_states: The proof states used to predict the tactics for.
        :param tactics_so_far: Optional list of tactics so far
        :param previous_proof_states: Optional list of previous proof states
        :param temperature: The temperature to use, 0 for greedy sampling
        :param max_new_tokens: The maximum number of new tokens to generate for the tactics
        :return: The subsequent tactics
        """
        output_single = False
        if isinstance(proof_states, str):
            output_single = True
            proof_states = [proof_states]
            assert not tactics_so_far or isinstance(tactics_so_far[0], str)
            tactics_so_far = [tactics_so_far] if tactics_so_far is not None else None
            assert not previous_proof_states or isinstance(previous_proof_states[0], str)
            previous_proof_states = [previous_proof_states] if previous_proof_states is not None else None
        prompts = [self._build_prompt(proof_state, tactics_so_far, previous_proof_states) for proof_state in
                   proof_states]
        prompt_results = self.tokenizer.pad({"input_ids": prompts}, padding_side="right", return_attention_mask=True,
                                            return_tensors="pt")
        prompt_tensor = prompt_results.input_ids.to(self.device)
        tokens = None
        tactics = []
        idx = 0
        while tokens is None or (tokens != self.eos_token).any() and idx < max_new_tokens:
            logits = self.model(prompt_tensor)[:, -1, ...]
            if temperature > 0.0:
                softmaxed = self.softmax(logits / temperature)
                tokens = torch.multinomial(softmaxed, 1).squeeze(1)
            else:
                tokens = logits.argmax(dim=1)
            prompt_tensor = torch.cat([prompt_tensor, tokens[:, None]], dim=1)
            tactics.append(tokens)
            idx += 1
        tactics = torch.stack(tactics, dim=1)
        tactics = tactics[:, :-1]
        tactics = tactics.tolist()
        result_ids = []
        for tactic in tactics:
            result = []
            for token in tactic:
                if token == self.eos_token:
                    break
                result.append(token)
            result_ids.append(result)
        if output_single:
            return self.tokenizer.decode(result_ids[0])
        return self.tokenizer.batch_decode(result_ids)

    def next_tactics(self, proof_state: str, k: int, tactics_so_far: Optional[List[str]] = None,
                     temperature: float = 0.0, max_new_tokens: int = 20) -> List[str]:
        """Predict the subsequent tactics for the given proof state (which might have multiple goals)

        :param proof_state: The proof state used to predict the tactics for.
        :param k: The number of tactics to predict
        :param tactics_so_far: Optional list of tactics so far
        :param temperature: The temperature to use, 0 for greedy sampling
        :param max_new_tokens: The maximum number of new tokens to generate for the tactics
        :return: The subsequent tactics
        """
        prompt = self._build_prompt(proof_state, tactics_so_far)
        prompt_tensor = torch.tensor(prompt).to(self.device).repeat(k, 1)
        tokens = None
        tactics = []
        idx = 0
        while tokens is None or (tokens != self.eos_token).any() and idx < max_new_tokens:
            logits = self.model(prompt_tensor)[:, -1, ...]
            if temperature > 0.0:
                softmaxed = self.softmax(logits / temperature)
                tokens = torch.multinomial(softmaxed, 1).squeeze(1)
            else:
                tokens = logits.argmax(dim=1)
            prompt_tensor = torch.cat([prompt_tensor, tokens[:, None]], dim=1)
            tactics.append(tokens)
            idx += 1
        tactics = torch.stack(tactics, dim=1)
        tactics = tactics[:, :-1]
        tactics = tactics.tolist()

        result_ids = [[] for _ in range(len(proof_states))]
        for idx, tactic in enumerate(tactics):
            result = []
            for token in tactic:
                if token == self.eos_token:
                    break
                result.append(token)
            result_ids[idx % len(proof_states)].append(result)
        if output_single:
            return self.tokenizer.batch_decode(result_ids[0])
        return [self.tokenizer.batch_decode(result) for result in result_ids]

    def _build_prompt(self, proof_state: str, tactics_so_far: Optional[List[str]] = None,
                      proof_states_so_far: Optional[List[str]] = None) -> List[int]:
        state_ids: List[int] = self.tokenizer.encode(proof_state)
        to_append = []
        if tactics_so_far is not None:
            tactics: List[List[int]] = [self.tokenizer.encode(tactic) for tactic in tactics_so_far]
            for idx, tactic in enumerate(tactics):
                if idx < len(tactics) - 1:
                    tactic.append(self.tactics_sep_id)
            tactics_flat: List[int] = sum(tactics, [])
            to_append = [self.tactics_id] + tactics_flat
        if proof_states_so_far is not None:
            proof_states: List[List[int]] = [self.tokenizer.encode(proof_state) + [self.proofstate_sep_id] for
                                             proof_state in proof_states_so_far]
            proof_states_flat: List[int] = sum(proof_states, [])
            state_ids = proof_states_flat + state_ids
        return [self.proof_state_id] + state_ids + to_append + [self.proof_step_id]

    def _forward(self, batch: BatchEncoding) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask[:, 1:]
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        logits = self.model(input_ids)[:, :-1, :]
        labels = input_ids[:, 1:]
        labels = labels.masked_fill(~attention_mask.bool(), -100)
        return logits, labels

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
        logits, labels = self._forward(padded)
        # Only compute loss for the part after the prompt
        if not loss_on_prompt:
            for i in range(len(prompts)):
                labels[i, :len(prompts[i]) - 1] = -100
        return self.loss_fn(logits.transpose(2, 1), labels)

    def evaluate_batch(self, batch: list[TrainingSample]) -> dict[str, float]:
        """Evaluate on one single batch of training samples.

        :param batch: The batch to evaluate on
        :return: The metrics
        """
        prompts = [self._build_prompt(sample.proof_state, sample.tactics_so_far) for sample in batch]
        tactics = [self.tokenizer.encode(sample.tactic) for sample in batch]

        full = {
            "input_ids": [prompt + tactic + [self.eos_token] for prompt, tactic in zip(prompts, tactics, strict=True)]}

        padded = self.tokenizer.pad(full, padding_side="right", return_attention_mask=True, return_tensors="pt")
        logits, labels = self._forward(padded)
        # Only compute loss for the part after the prompt
        for i in range(len(prompts)):
            labels[i, :len(prompts[i]) - 1] = -100
        loss = self.loss_fn(logits.transpose(2, 1), labels)
        # Exact match
        is_correct: torch.Tensor = logits.argmax(dim=-1) == labels
        # Ignore padding
        accuracy = (is_correct.sum().item() / (is_correct.numel() - (labels == -100).sum())).item()
        return {"loss": loss.item(), "perplexity": loss.exp().item(), "accuracy": accuracy}

    def save(self, path: str | Path):
        state_dict = self.model.state_dict()
        result = {"state_dict": state_dict, "eos_id": self.eos_token, "proof_step_id": self.proof_step_id,
                  "proof_state_id": self.proof_state_id, "tactics_id": self.tactics_id,
                  "tactics_sep_id": self.tactics_sep_id}
        torch.save(result, path)

    def load(self, path: str | Path):
        result = torch.load(path)
        self.eos_token = result["eos_id"]
        self.proof_step_id = result["proof_step_id"]
        self.proof_state_id = result["proof_state_id"]
        self.tactics_id = result["tactics_id"]
        self.tactics_sep_id = result["tactics_sep_id"]
        self.proofstate_sep_id = result["proofstate_sep_id"]
        self.goals_sep_id = result["goals_sep_id"]
        self.successful_proof_token = result["successful_proof_token"]
        self.incomplete_proof_token = result["incomplete_proof_token"]
        self.invalid_proof_token = result["invalid_proof_token"]
        self.model.load_state_dict(result["state_dict"])
        self.model.eval()

    @classmethod
    def from_file(cls, path: str | Path, model: nn.Module, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                  device: str = "cpu"):
        result = torch.load(path)
        model.load_state_dict(result["state_dict"])
        model.to(device)
        model.eval()
        return cls(model, result["eos_id"], result["proof_step_id"], result["proof_state_id"], result["tactics_id"],
                   result["tactics_sep_id"], result["proofstate_sep_id"], result["goals_sep_id"],
                   result["successful_proof_token"], result["incomplete_proof_token"], result["invalid_proof_token"],
                   tokenizer, device)


class MambaPolicy(Policy):

    def __init__(self, model: nn.Module, mamba_config: MambaConfig,
                 eos_id: int, proof_step_id: int, proof_state_id: int, tactics_id: int,
                 tactics_sep_id: int, proofstate_sep_id: int, goals_sep_id: int, successful_proof_token: int,
                 incomplete_proof_token: int, invalid_proof_token: int,
                 tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                 device: str = "cpu"):
        super().__init__(model, eos_id, proof_step_id, proof_state_id, tactics_id, tactics_sep_id, goals_sep_id,
                         proofstate_sep_id,
                         successful_proof_token, incomplete_proof_token, invalid_proof_token, tokenizer, device)
        self.config = mamba_config

    def load(self, path: str | Path):
        super().load(path)
        result = torch.load(path)
        config_dict = result["config"]
        self.config = MambaConfig(**config_dict)

    def save(self, path: str | Path):
        state_dict = self.model.state_dict()
        result = {"state_dict": state_dict, "eos_id": self.eos_token, "proof_step_id": self.proof_step_id,
                  "proof_state_id": self.proof_state_id, "tactics_id": self.tactics_id,
                  "tactics_sep_id": self.tactics_sep_id,
                  "proofstate_sep_id": self.proofstate_sep_id, "goals_sep_id": self.goals_sep_id,
                  "successful_proof_token": self.successful_proof_token,
                  "incomplete_proof_token": self.incomplete_proof_token,
                  "invalid_proof_token": self.invalid_proof_token, "config": asdict(self.config)}
        torch.save(result, path)

    @classmethod
    def from_file(cls, path: str | Path, mamba_config: MambaConfig, eos_id: int, proof_step_id: int,
                  proof_state_id: int, tactics_id: int, tactics_sep_id: int, proofstate_sep_id: int, goals_sep_id: int,
                  successful_proof_token: int, incomplete_proof_token: int, invalid_proof_token: int,
                  is_gflownet: bool, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, device: str = "cpu"):
        model = MambaLMHeadModelWrapper(mamba_config, device=device, is_gflownet=is_gflownet)
        model.load_state_dict(torch.load(path, map_location=device))
        return cls(model, mamba_config, eos_id, proof_step_id, proof_state_id, tactics_id, tactics_sep_id,
                   proofstate_sep_id,
                   goals_sep_id, successful_proof_token, incomplete_proof_token, invalid_proof_token, tokenizer, device)
