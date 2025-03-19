from proofflow.train_gfn import MCTS, Node, train_gflownet, sample_mcts_trajectories, _compute_log_internlm
from lean_repl_py import LeanREPLHandler, LeanREPLProofState, LeanREPLNextProofState
from pathlib import Path
import torch
import pytest

@pytest.fixture
def handler():
    yield LeanREPLHandler(Path("./leanproject"))

@pytest.fixture
def monkeypatch_torch(monkeypatch):
    monkeypatch.setattr("torch.Tensor.backward", lambda *args, **kwargs: None)


class MockModel:
    def __init__(self):
        pass

    def train(self):
        pass

    def log_z(self, *args, **kwargs):
        return torch.tensor(0.0)

    def p_b(self, x):
        return self(x)

    def __call__(self, x):
        return torch.rand(1, x.shape[1], 10)

    def parameters(self):
        return []


class MockEncoding:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 0

    def encode(self, *args, **kwargs):
        return [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def pad(self, input_dict, **kwargs):
        return_tensors = kwargs.get('return_tensors', 'pt')
        assert "input_ids" in input_dict
        input_ids = input_dict["input_ids"]
        if return_tensors == 'pt':
            input_tensors = torch.tensor(input_ids)
            return MockEncoding(input_tensors, torch.ones_like(input_tensors))
        return input_ids


class MockPolicy:
    def __init__(self, tactics: list[list[list[str]]]):
        self.model = MockModel()
        self.tactics = tactics
        self.tactic_idx = 0
        self.tokenizer = MockTokenizer()
        self.successful_proof_token = 1
        self.invalid_proof_token = 2
        self.proofstate_sep_id = 3
        self.proof_step_id = 4

    def next_tactics_int(self, *args, **kwargs):
        if self.tactic_idx < len(self.tactics):
            tactic = self.tactics[self.tactic_idx]
            self.tactic_idx += 1
            return tactic, None, None
        else:
            return [], None, None

    def next_tactics(self, *args, **kwargs):
        return self.next_tactics_int(*args, **kwargs)[0]

    def _build_prompt(self, *args, **kwargs):
        return list(range(1, 10))


class MockOptimizer:
    def __init__(self, *args, **kwargs):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def test_simple_proof():
    handler = LeanREPLHandler()
    handler.send_command("""theorem p_and_q (p q : Prop) (a : p) (b : q): p ∧ q := by
        sorry""")
    response, env = handler.receive_json()
    proof_state = response["sorries"][0]
    assert isinstance(proof_state, LeanREPLProofState)
    root = Node(proof_state.goal, proof_state_idx=proof_state.proof_state)
    mcts = MCTS(root)
    next_node = mcts.select()
    assert next_node == root
    handler.send_tactic("constructor", proof_state.proof_state)
    response, env = handler.receive_json()
    assert isinstance(response, LeanREPLNextProofState)
    root.expand(["constructor"], [response.goals[0]], [0.3], [-3], [response.proof_state])
    next_node = mcts.select()
    assert next_node == root.children_for_tactics["constructor"]

    handler.send_tactic("exact a", response.proof_state)
    response, env = handler.receive_json()
    assert isinstance(response, LeanREPLNextProofState)
    next_node.expand(["exact a"], [response.goals[0]], [0.3], [-3], [response.proof_state])
    next_node = mcts.select()
    assert next_node == root.children_for_tactics["constructor"].children_for_tactics["exact a"]


def test_simple_monkeypatch(monkeypatch, monkeypatch_torch, handler):
    handler_factory = lambda: handler
    handler.send_command("""
import Mathlib.Data.Matrix.Basic
open Matrix
variable {l m n o p q : Type*} {m' n' p' : o → Type*}
variable {R : Type*} {S : Type*} {α : Type*} {β : Type*}
variable [DecidableEq o]
variable [Zero α] [Zero β]
def blockDiagonal' (M : ∀ i, Matrix (m' i) (n' i) α) : Matrix (Σi, m' i) (Σi, n' i) α :=
  of <|
    (fun ⟨k, i⟩ ⟨k', j⟩ => if h : k = k' then M k i (cast (congr_arg n' h.symm) j) else 0 :
      (Σi, m' i) → (Σi, n' i) → α)

theorem blockDiagonal'_apply (M : ∀ i, Matrix (m' i) (n' i) α) (ik jk) :
    blockDiagonal' M ik jk =
      if h : ik.1 = jk.1 then M ik.1 ik.2 (cast (congr_arg n' h.symm) jk.2) else 0 := by sorry""")
    response, env = handler.receive_json()
    proof_state = response["sorries"][0]
    assert isinstance(proof_state, LeanREPLProofState)
    root = Node(proof_state.goal, proof_state_idx=proof_state.proof_state)
    mcts = MCTS(root)
    monkeypatch.setattr("proofflow.train_gfn._get_start_states", lambda *args, **kwargs: ([proof_state], [mcts]))
    policy = MockPolicy([[['by_cases h : k i = n',
                           ', , rfl h , ( _erase _zero ] ₂, ℕ ] ( ort vec _apply map_map mul_one _singleton',
                           'cases i k',
                           "_eq_prod_ , _ inter_comm , _self sup_sdiff _self , ← _def ' ( group smul_eq_mul _apply range max coe",
                           'obtain rfl | hij := Decidable . eq_or_ne i k']],
                         [['cases h', 'obtain h | h | h := h', 'cases i k', "simp only [ block Di ag ' _apply ]",
                           'rcases i k with ⟨⟩']],
                         [['simp only [ lt_irrefl ]', "simp_rw [ block Di ag ' _apply ]", 'simp', 'simp', 'simp']],
                         [['simp', 'simp', 'simp', 'simp', 'simp']]])
    train_gflownet(policy, [], [1], handler_factory, Path("./mathlib4"), MockOptimizer(), MockOptimizer(), 5, 1, 0, 30,
                   "cpu")


def test_mcts_simple_monkeypatch_failure(monkeypatch, handler, monkeypatch_torch):
    handler.send_command("""theorem pandq (p q : Prop) (a : p) (b : q) : p ∧ q := by sorry""")
    response, env = handler.receive_json()
    proof_state = response["sorries"][0]
    assert isinstance(proof_state, LeanREPLProofState)
    root = Node(proof_state.goal, proof_state_idx=proof_state.proof_state)
    mcts = MCTS(root)
    monkeypatch.setattr("proofflow.train_gfn._get_start_states", lambda *args, **kwargs: ([proof_state], [mcts]))

    policy = MockPolicy([[["exact a", "exact b", "constructor", "linarith", "apply And.intro"]],
                          [["exact a", "constructor", "exact b", "something1", "something2"]],
                          [["exact c", "exact d", "exact m", "exact n", "something3"]],
                          [["exact c", "exact d", "exact m", "exact n", "something3"]]])
    states, actions, rewards, nodes_proven, valid_ratio = sample_mcts_trajectories(policy, [mcts], [handler], 50, "cuda", 10, 5, 0.7)
    assert states == [[[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 3, 2, 4]]]
    assert actions == [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]]
    assert pytest.approx(rewards[0], abs=1e-7) == _compute_log_internlm([False], [True], [None], 3, [None])[0]
    assert nodes_proven == 0
    assert valid_ratio == 0.5

def test_mcts_simple_monkeypatch_working(monkeypatch, handler, monkeypatch_torch):
    handler.send_command("""theorem pandq (p q : Prop) (a : p) (b : q) : p ∧ q := by sorry""")
    response, env = handler.receive_json()
    proof_state = response["sorries"][0]
    assert isinstance(proof_state, LeanREPLProofState)
    root = Node(proof_state.goal, proof_state_idx=proof_state.proof_state)
    mcts = MCTS(root)
    monkeypatch.setattr("proofflow.train_gfn._get_start_states", lambda *args, **kwargs: ([proof_state], [mcts]))

    policy = MockPolicy([[["exact a", "exact b", "constructor", "linarith", "apply And.intro"]],
                          [["exact a", "constructor", "exact b", "something1", "something2"]],
                          [["exact c", "exact d", "exact m", "exact n", "something3"]],
                          [["exact b", "exact d", "exact m", "exact n", "somethming3"]]])
    states, actions, rewards, nodes_proven, valid_ratio = sample_mcts_trajectories(policy, [mcts], [handler], 50, "cuda", 10, 5, 0.7)
    assert states == [[[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9],[1, 2, 3, 4, 5, 6, 7, 8, 3, 1, 4]]]
    assert actions == [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]]
    assert pytest.approx(rewards[0], abs=1e-3) == pytest.approx(_compute_log_internlm([True], [None], [None], 4, [None])[0], abs=1e-3)
    assert nodes_proven == 1
    assert valid_ratio == 0.75


def test_mcts_simple_monkeypatch_working_advanced(monkeypatch, handler, monkeypatch_torch):
    handler.send_command("""theorem pandq (p q : Prop) (a : p) (b : q) : p ∧ q := by sorry""")
    response, env = handler.receive_json()
    proof_state = response["sorries"][0]
    assert isinstance(proof_state, LeanREPLProofState)
    root = Node(proof_state.goal, proof_state_idx=proof_state.proof_state)
    mcts = MCTS(root)
    monkeypatch.setattr("proofflow.train_gfn._get_start_states", lambda *args, **kwargs: ([proof_state], [mcts]))

    policy = MockPolicy([[["exact a", "exact b", "constructor", "linarith", "apply And.intro"]],
                          [["exact a", "constructor", "exact b", "something1", "something2"]],
                          [["exact c", "exact d", "exact m", "exact n", "something3"]],
                          [["exact b", "apply And.left", "exact m", "exact n", "somethming3"]],
                          [["exact b", "constructor", "exact m", "exact n", "something3"]],
                          [["exact g", "exact b", "exact m", "exact n", "somethming3"]],
                          [["exact g", "simpa", "exact m", "exact n", "somethming3"]]])
    states, actions, rewards, nodes_proven, valid_ratio =  sample_mcts_trajectories(policy, [mcts], [handler], 50, "cuda", 10, 5, 0.7)
    assert states == [[[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9],[1, 2, 3, 4, 5, 6, 7, 8, 3, 1, 4]]]
    assert actions == [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]]
    assert pytest.approx(rewards[0], abs=1e-3) == pytest.approx(_compute_log_internlm([True], [None], [None], 4, [None])[0], abs=1e-3)
    assert nodes_proven == 1
    assert pytest.approx(valid_ratio, abs=1e-4) == 0.85714