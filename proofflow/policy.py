from torch import nn
from typing import Optional, List, Union, Tuple
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding, AutoTokenizer, AutoModel, T5Model, \
    AutoModelForSeq2SeqLM
import torch
from proofflow.data import TrainingSample
from pathlib import Path
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm import MambaLMHeadModel
from dataclasses import asdict
import warnings


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

    def log_z(self, input_ids, attention_mask):
        hidden_states = self.backbone(input_ids)
        # Get the last one that has attention one
        last_indices = attention_mask.sum(1) - 1
        hidden_states = hidden_states[torch.arange(hidden_states.shape[0]), last_indices]
        lm_logits = self.z_head(hidden_states)
        return lm_logits

    def get_non_z_params(self):
        return [i for i in self.parameters() if all(id(i) != id(j) for j in self.get_z_params())]

    def get_z_params(self):
        return self.z_head.parameters()


class ReProver(nn.Module):
    def __init__(self, model: T5Model, *args, device=None, dtype=None, is_gflownet: bool = False, **kwargs):
        super().__init__()
        self.backbone = model
        if is_gflownet:
            self.back_head = nn.Linear(model.decoder.config.d_model, model.config.vocab_size, bias=False, device=device,
                                       dtype=dtype)
            self.z_head = nn.Linear(model.decoder.config.d_model, 1, bias=True, device=device, dtype=dtype)
        else:
            self.back_head, self.z_head = None, None

    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)

    def forward(self, input_ids, attention_mask, decoder_input_ids = None, decoder_attention_mask = None):
        if decoder_input_ids is None:
            decoder_input_ids = self.backbone._shift_right(input_ids)
        return self.backbone(input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask,
                             decoder_input_ids=decoder_input_ids).logits

    def p_b(self, input_ids):
        decoder_input_ids = self.backbone._shift_right(input_ids)
        hidden_states = self.backbone(input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
        back_logits = self.back_head(hidden_states)
        return back_logits

    def log_z(self, input_ids, attention_mask):
        decoder_input_ids = self.backbone._shift_right(input_ids)
        hidden_states = self.backbone(input_ids, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
        # Get the last one that has attention one
        last_indices = attention_mask.sum(1) - 1
        hidden_states = hidden_states[torch.arange(hidden_states.shape[0]), last_indices]
        lm_logits = self.z_head(hidden_states)
        return lm_logits

    def get_non_z_params(self):
        return [i for i in self.parameters() if all(id(i) != id(j) for j in self.get_z_params())]

    def get_z_params(self):
        return self.z_head.parameters()


class Policy:
    """
    A policy generating subsequent tactics given a proof state and optionally the tactics so far

    Will predict tactics querying the model using the prompt
    PROOFSTATE <state> PROOFSTEP

    Optionally, if tactics so far are given, will instead prompt as
    PROOFSTATE <state> TACTICS <tactics> PROOFSTEP
    """

    def __init__(self, model: nn.Module, eos_id: int, proof_step_id: int, proof_state_id: int, tactics_id: int,
                 tactics_sep_id: int, proofstate_sep_id: int, successful_proof_token: int,
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
        self.successful_proof_token = successful_proof_token
        self.incomplete_proof_token = incomplete_proof_token
        self.invalid_proof_token = invalid_proof_token
        self.tokenizer = tokenizer
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device

    def next_tactic(self, proof_states: Union[List[str], str],
                    tactics_so_far: Optional[Union[List[List[str]], List[str]]] = None,
                    previous_proof_states: Optional[Union[List[List[str]], List[str]]] = None,
                    temperature: float = 0.0, max_new_tokens: int = 20, state_skip: int = 1) -> Union[List[str], str]:
        """Predict the subsequent tactics for the given proof states (which might have multiple goals)
        This is a wrapper around next_tactic_int, which also returns tokenization and prompts to avoid
        recomputing these when training the GFlowNet.

        :param proof_states: The proof states used to predict the tactics for.
        :param tactics_so_far: Optional list of tactics so far
        :param previous_proof_states: Optional list of previous proof states
        :param temperature: The temperature to use, 0 for greedy sampling
        :param max_new_tokens: The maximum number of new tokens to generate for the tactics
        :return: The subsequent tactics
        """
        return self.next_tactic_int(proof_states, tactics_so_far, previous_proof_states, temperature, max_new_tokens, state_skip=state_skip)[0]

    def next_tactic_int(self, proof_states: Union[List[str], str],
                        tactics_so_far: Optional[Union[List[List[str]], List[str]]] = None,
                        previous_proof_states: Optional[Union[List[List[str]], List[str]]] = None,
                        temperature: float = 0.0, max_new_tokens: int = 20, state_skip: int = 1) -> \
                            Tuple[Union[List[str], str], Union[List[List[int]], List[int]], Union[List[List[int]], List[int]]]:
        """Predict the subsequent tactics for the given proof states (which might have multiple goals)
        Additionally, return the tokenization and prompts generated.

        :param proof_states: The proof states used to predict the tactics for.
        :param tactics_so_far: Optional list of tactics so far
        :param previous_proof_states: Optional list of previous proof states
        :param temperature: The temperature to use, 0 for greedy sampling
        :param max_new_tokens: The maximum number of new tokens to generate for the tactics
        :return: The subsequent tactics, the tokenization of the tactics, the generated prompts
        """
        tactics, tactic_ids, prompts = self.next_tactics_int(proof_states, 1, tactics_so_far, previous_proof_states, temperature,
                                             max_new_tokens, state_skip=state_skip)
        assert len(tactics) == len(tactic_ids)
        if len(tactics) == 1:
            return tactics[0], tactic_ids[0], prompts
        return tactics, tactic_ids, prompts

    def next_tactics(self, proof_states: Union[List[str], str], k: int,
                     tactics_so_far: Optional[Union[List[List[str]], List[str]]] = None,
                     previous_proof_states: Optional[Union[List[List[str]], List[str]]] = None,
                     temperature: float = 0.0, max_new_tokens: int = 20, state_skip: int = 1) -> Union[List[List[str]], List[str]]:
        """Predict the subsequent tactics for the given proof states (which might have multiple goals)
        This is a wrapper around next_tactics_int, which also returns tokenization and prompts to avoid
        recomputing these when training the GFlowNet.

        :param proof_states: The proof states used to predict the tactics for.
        :param k: The number of tactics to predict
        :param tactics_so_far: Optional list of tactics so far
        :param previous_proof_states: Optional list of previous proof states
        :param temperature: The temperature to use, 0 for greedy sampling
        :param max_new_tokens: The maximum number of new tokens to generate for the tactics
        :return: The subsequent tactics
        """
        return self.next_tactics_int(proof_states, k, tactics_so_far, previous_proof_states, temperature, max_new_tokens, state_skip=state_skip)[0]

    def next_tactics_int(self, proof_states: Union[List[str], str], k: int,
                     tactics_so_far: Optional[Union[List[List[str]], List[str]]] = None,
                     previous_proof_states: Optional[Union[List[List[str]], List[str]]] = None,
                     temperature: float = 0.0, max_new_tokens: int = 20, state_skip: int = 1) -> \
                            Tuple[Union[List[List[str]], List[str]], Union[List[List[List[int]]], List[List[int]]], Union[List[List[int]], List[int]]]:
        """Predict the subsequent tactics for the given proof states (which might have multiple goals)
        Additionally, return the tokenization and prompts generated.

        :param proof_states: The proof states used to predict the tactics for.
        :param k: The number of tactics to predict
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

        if not tactics_so_far:
            tactics_so_far = [None] * len(proof_states)
        if not previous_proof_states:
            previous_proof_states = [None] * len(proof_states)

        assert len(proof_states) == len(tactics_so_far) == len(previous_proof_states)

        prompts = [self._build_prompt(proof_state, preceding_tactics, preceding_states, state_skip=state_skip)
                    for proof_state, preceding_tactics, preceding_states in
                    zip(proof_states, tactics_so_far, previous_proof_states)]
        prompt_results = self.tokenizer.pad({"input_ids": prompts}, padding_side="right", return_tensors="pt")
        # Will repeat the prompt k times for each proof state one by one
        prompt_tensor = prompt_results.input_ids.to(self.device)[None].repeat(k, 1, 1).transpose(0, 1).reshape(-1,
                                                                                                               prompt_results.input_ids.shape[
                                                                                                                   1])
        tactics = []
        idx = 0
        eos = torch.zeros(prompt_tensor.shape[0], dtype=torch.bool, device=self.device)
        last_index = (prompt_results.attention_mask.sum(1) - 1).repeat(k, 1).view(-1)
        while not eos.all() and idx < max_new_tokens:
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                # Only take the last token for sampling next tokens, classical language model
                logits = self.model(prompt_tensor)[torch.arange(prompt_tensor.shape[0]), last_index, :]
            if temperature > 0.0:
                softmaxed = self.softmax(logits / temperature)
                tokens = torch.multinomial(softmaxed, 1).squeeze(1)
            else:  # if we take argmax, presumably we may as well just call next_tactic
                if k > 1:
                    warnings.warn("Taking argmax in next_tactics_int, this is equivalent to calling next_tactic")
                tokens = logits.argmax(dim=1)
            eos |= tokens == self.eos_token
            prompt_tensor = torch.cat([prompt_tensor, tokens[:, None]], dim=1)
            tactics.append(tokens)
            idx += 1
            last_index += 1
        tactics = torch.stack(tactics, dim=1)
        tactics = tactics[:, :-1] # Remove the last token, which is the eos token
        tactics = tactics.reshape(len(proof_states), k, tactics.shape[1])
        tactics = tactics.tolist()

        result_ids = [[] for _ in range(len(proof_states))]
        for proof_state_idx in range(len(tactics)):
            for tactic in tactics[proof_state_idx]:
                result = []
                for token in tactic:
                    if token == self.eos_token:
                        break
                    result.append(token)
                result_ids[proof_state_idx].append(result)
        if output_single:
            return self.tokenizer.batch_decode(result_ids[0]), result_ids[0], prompts[0]
        return [self.tokenizer.batch_decode(result) for result in result_ids], result_ids, prompts

    def _build_prompt(self, proof_state: str, tactics_so_far: Optional[List[str]] = None,
                      proof_states_so_far: Optional[List[str]] = None, state_skip: int = 10) -> List[int]:
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
            proof_states_so_far = proof_states_so_far[len(proof_states_so_far) % state_skip :: state_skip]
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

    def train_batch(self, batch: list[TrainingSample], loss_on_prompt: bool = False, tactics_so_far: bool = False,
                    proof_states_so_far: bool = False, state_skip: int = 1) -> torch.Tensor:
        """Train on one single batch of training samples.

        :param batch: The batch to train on
        :param loss_on_prompt: Whether to also compute language modelling loss on prompt tokens.
        :param tactics_so_far: Whether to condition on tactics so far
        :param proof_states_so_far: Whether to condition on proof states so far
        :return:
        """
        if tactics_so_far and proof_states_so_far:
            prompts = [self._build_prompt(sample.proof_state, sample.tactics_so_far, sample.proof_states_so_far, state_skip=state_skip) for
                       sample in batch]
        elif tactics_so_far:
            prompts = [self._build_prompt(sample.proof_state, sample.tactics_so_far, state_skip=state_skip) for sample in batch]
        elif proof_states_so_far:
            prompts = [self._build_prompt(sample.proof_state, proof_states_so_far=sample.proof_states_so_far, state_skip=state_skip) for sample
                       in batch]
        else:
            prompts = [self._build_prompt(sample.proof_state, state_skip=state_skip) for sample in batch]
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

    def evaluate_batch(self, batch: list[TrainingSample], state_skip : int = 1) -> dict[str, float]:
        """Evaluate on one single batch of training samples.

        :param batch: The batch to evaluate on
        :return: The metrics
        """
        prompts = [self._build_prompt(sample.proof_state, sample.tactics_so_far, state_skip=state_skip) for sample in batch]
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
                  "tactics_sep_id": self.tactics_sep_id, "proofstate_sep_id": self.proofstate_sep_id,
                  "successful_proof_token": self.successful_proof_token,
                  "incomplete_proof_token": self.incomplete_proof_token,
                  "invalid_proof_token": self.invalid_proof_token}
        torch.save(result, path)

    def load(self, path: str | Path):
        result = torch.load(path)
        self.eos_token = result["eos_id"]
        self.proof_step_id = result["proof_step_id"]
        self.proof_state_id = result["proof_state_id"]
        self.tactics_id = result["tactics_id"]
        self.tactics_sep_id = result["tactics_sep_id"]
        self.proofstate_sep_id = result["proofstate_sep_id"]
        self.successful_proof_token = result["successful_proof_token"]
        self.incomplete_proof_token = result["incomplete_proof_token"]
        self.invalid_proof_token = result["invalid_proof_token"]
        self.model.load_state_dict(result["state_dict"])
        self.model.eval()

    @classmethod
    def from_file(cls, path: str | Path, model: nn.Module, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                  device: str = "cpu", strict: bool = True):
        result = torch.load(path)
        model.load_state_dict(result["state_dict"], strict=strict)
        model.to(device)
        model.eval()
        return cls(model, result["eos_id"], result["proof_step_id"], result["proof_state_id"], result["tactics_id"],
                   result["tactics_sep_id"], result["proofstate_sep_id"], result["successful_proof_token"],
                   result["incomplete_proof_token"], result["invalid_proof_token"], tokenizer, device)


class MambaPolicy(Policy):

    def __init__(self, model: nn.Module, eos_id: int, proof_step_id: int, proof_state_id: int, tactics_id: int,
                 tactics_sep_id: int, proofstate_sep_id: int, successful_proof_token: int,
                 incomplete_proof_token: int, invalid_proof_token: int,
                 tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                 device: str = "cpu", mamba_config: MambaConfig | None = None):
        super().__init__(model, eos_id, proof_step_id, proof_state_id, tactics_id, tactics_sep_id, proofstate_sep_id,
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
                  "tactics_sep_id": self.tactics_sep_id, "proofstate_sep_id": self.proofstate_sep_id,
                  "successful_proof_token": self.successful_proof_token,
                  "incomplete_proof_token": self.incomplete_proof_token,
                  "invalid_proof_token": self.invalid_proof_token, "config": asdict(self.config)}
        torch.save(result, path)

    @classmethod
    def from_file(cls, path: str | Path, is_gflownet: bool,
                  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, device: str = "cpu"):
        result = torch.load(path)
        config = result["config"]
        mamba_config = MambaConfig(**config)
        model = MambaLMHeadModelWrapper(mamba_config, device=device, is_gflownet=is_gflownet)
        # If gflownet, then we have uninitialized backhead, zhead
        result = super().from_file(path, model, tokenizer, device, strict=not is_gflownet)
        result.config = mamba_config
        return result


class ReProverPolicy(Policy):
    def __init__(self, model: nn.Module, eos_id: int, proof_step_id: int, proof_state_id: int,
                 tactics_id: int, tactics_sep_id: int, proofstate_sep_id: int,
                 successful_proof_token: int, incomplete_proof_token: int, invalid_proof_token: int,
                 tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
                 device: str = "cpu"):
        super().__init__(model, eos_id, proof_step_id, proof_state_id, tactics_id, tactics_sep_id, proofstate_sep_id,
                         successful_proof_token, incomplete_proof_token, invalid_proof_token, tokenizer, device)

    def _build_prompt(self, proof_state: str, tactics_so_far: Optional[List[str]] = None,
                      proof_states_so_far: Optional[List[str]] = None, state_skip: int = 10) -> List[int]:
        if proof_states_so_far is not None:
            proof_states_so_far = proof_states_so_far[len(proof_states_so_far) % state_skip:: state_skip]
            full = proof_states_so_far + [proof_state]
        else:
            full = [proof_state]
        return self.tokenizer.encode("\n".join(full))

    @classmethod
    def from_pretrained(cls, device: str, is_gflownet: bool) -> "ReProverPolicy":
        url = "kaiyuy/leandojo-lean4-tacgen-byt5-small"
        tokenizer = AutoTokenizer.from_pretrained(url)
        eos_id = 1
        proof_step_id = 259
        proof_state_id = 260
        tactics_id = 261
        tactics_sep_id = 262
        proofstate_sep_id = 263
        successful_proof_token = 264
        incomplete_proof_token = 265
        invalid_proof_token = 266
        model = AutoModelForSeq2SeqLM.from_pretrained(url)
        model.to(device)
        reprover = ReProver(model, device=device, is_gflownet=is_gflownet)
        return cls(reprover, eos_id, proof_step_id, proof_state_id, tactics_id, tactics_sep_id, proofstate_sep_id,
                   successful_proof_token, incomplete_proof_token, invalid_proof_token, tokenizer, device)

    def next_tactics_int(self, proof_states: Union[List[str], str], k: int,
                         tactics_so_far: Optional[Union[List[List[str]], List[str]]] = None,
                         previous_proof_states: Optional[Union[List[List[str]], List[str]]] = None,
                         temperature: float = 0.0, max_new_tokens: int = 20,  state_skip: int = 1) -> \
            Tuple[Union[List[List[str]], List[str]], Union[List[List[List[int]]], List[List[int]]], Union[
                List[List[int]], List[int]]]:
        output_single = False
        if isinstance(proof_states, str):
            output_single = True
            proof_states = [proof_states]
            assert not tactics_so_far or isinstance(tactics_so_far[0], str)
            tactics_so_far = [tactics_so_far] if tactics_so_far is not None else None
            assert not previous_proof_states or isinstance(previous_proof_states[0], str)
            previous_proof_states = [previous_proof_states] if previous_proof_states is not None else None

        if not tactics_so_far:
            tactics_so_far = [None] * len(proof_states)
        if not previous_proof_states:
            previous_proof_states = [None] * len(proof_states)

        assert len(proof_states) == len(tactics_so_far) == len(previous_proof_states)
        previous_proof_states = previous_proof_states[len(previous_proof_states) % state_skip:: state_skip]
        result_ids = []
        for proof_state, previous_states in zip(proof_states, previous_proof_states):
            prompt = "\n".join(proof_state)
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
            input_ids = prompt_tokens["input_ids"].to(self.device)
            attention_mask = prompt_tokens["attention_mask"].to(self.device)
            tactic_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, num_return_sequences=k, do_sample=True,
                                             temperature=temperature)
            result_ids.append(tactic_ids)

        prompts = [self._build_prompt(proof_state, preceding_tactics, preceding_states)
                   for proof_state, preceding_tactics, preceding_states in
                   zip(proof_states, tactics_so_far, previous_proof_states)]
        if output_single:
            return self.tokenizer.batch_decode(result_ids[0], skip_special_tokens=True), result_ids[0], prompts[0]
        return [self.tokenizer.batch_decode(result, skip_special_tokens=True) for result in result_ids], result_ids, prompts


    def logprobs(self, proof_states: Union[List[str], str], tactics: Union[List[str], str]) -> float:
        if isinstance(proof_states, str):
            proof_states = [proof_states]
            tactics = [tactics]
        assert len(proof_states) == len(tactics)
        full = [proof_state + "\n" + tactic for proof_state, tactic in zip(proof_states, tactics)]
        tokenized = self.tokenizer(full, return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        encoder_tokenized = self.tokenizer(proof_states, return_tensors="pt", padding=True)
        encoder_ids = encoder_tokenized.input_ids.to(self.device)
        encoder_mask = encoder_tokenized.attention_mask.to(self.device)
        logits = self.model(encoder_ids, encoder_mask, decoder_input_ids=input_ids, decoder_attention_mask=attention_mask)
        # Mask the proof state part
        max_len = input_ids.shape[1]
        proof_state_ids_list = self.tokenizer(proof_states, add_special_tokens=False).input_ids
        tactic_ids_list = self.tokenizer(tactics, add_special_tokens=False).input_ids
        labels = torch.tensor([[-100] * len(proof_state) + tactic + [-100] * (max_len - len(proof_state) - len(tactic)) for proof_state, tactic in zip(proof_state_ids_list, tactic_ids_list)], device=self.device)
        ce = torch.nn.functional.cross_entropy(logits.view(labels.shape[0], -1, labels.shape[1]), labels, reduction="sum")
        return -ce.item()

    def evaluate_batch(self, batch: list[TrainingSample], state_skip : int = 1) -> dict[str, float]:
        """Evaluate on one single batch of training samples.

        :param batch: The batch to evaluate on
        :return: The metrics
        """
        prompts = [self._build_prompt(sample.proof_state + "\n", sample.tactics_so_far, state_skip=state_skip)[:-1] for sample in batch]
        tactics = [self.tokenizer.encode(sample.tactic, add_special_tokens=False) for sample in batch]

        full = {
            "input_ids": [prompt + tactic + [self.eos_token] for prompt, tactic in zip(prompts, tactics, strict=True)]}

        padded = self.tokenizer.pad(full, padding_side="right", return_attention_mask=True, return_tensors="pt")
        encoder_ids = self.tokenizer.pad({"input_ids": [prompt + [self.eos_token] for prompt in prompts]}, padding_side="right", return_attention_mask=True, return_tensors="pt")

        encoder_input_ids = encoder_ids["input_ids"].to(self.device)
        encoder_attention_mask = encoder_ids["attention_mask"].to(self.device)
        decoder_input_ids = padded["input_ids"].to(self.device)
        decoder_attention_mask = padded["attention_mask"].to(self.device)

        logits = self.model(encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask)
        # Only compute loss for the part after the prompt
        labels = decoder_input_ids[:, 1:]
        attention_mask = decoder_attention_mask[:, 1:]
        labels = labels.masked_fill(~attention_mask.bool(), -100)
        for i in range(len(prompts)):
            labels[i, :len(prompts[i]) - 1] = -100
        logits = logits[:, :-1, :]
        loss = self.loss_fn(logits.transpose(2, 1), labels)
        # Exact match
        is_correct: torch.Tensor = logits.argmax(dim=-1) == labels
        # Ignore padding
        accuracy = (is_correct.sum().item() / (is_correct.numel() - (labels == -100).sum())).item()
        return {"loss": loss.item(), "perplexity": loss.exp().item(), "accuracy": accuracy}

