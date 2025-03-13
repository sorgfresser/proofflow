from pathlib import Path
import json
from typing import Any, Sequence, Generator
from pydantic import BaseModel, model_validator
from lean_repl_py import LeanREPLProofState, LeanREPLHandler
from torch.utils.data import Dataset
import re

LEAN_DOJO_PATH = Path("../leandojo_benchmark_4/random")

ATTRIBUTE_REGEX = re.compile(r"@\[\s*(?:[^\[\]]|\[(?:[^\]]|\[[^\]]*\])*\])*\]")
BY_REGEX = re.compile(r":=\s*by\s+")
WS = r"\s+"
MAX_WS = r"^\s*"


class UnknownMetaVariableError(RuntimeError):
    pass

class TrainingSample(BaseModel):
    proof_state: str
    tactic: str
    tactics_so_far: list[str]


class Position(BaseModel):
    line: int
    column: int

    @model_validator(mode="before")
    @classmethod
    def from_list(cls, data: Any) -> Any:
        if isinstance(data, Sequence):
            assert len(data) == 2
            data = {"line": data[0], "column": data[1]}
        return data


class Annotation(BaseModel):
    def_end_pos: Position
    def_path: str
    def_pos: Position
    full_name: str


class AnnotatedTactic(BaseModel):
    annotated_tac: str
    annotations: list[Annotation]

    @model_validator(mode="before")
    @classmethod
    def from_list(cls, data: Any) -> Any:
        if isinstance(data, Sequence):
            assert len(data) == 2
            data = {"annotated_tac": data[0], "annotations": data[1]}
        return data


class TracedTactic(BaseModel):
    tactic: str
    annotated_tactic: AnnotatedTactic
    state_before: str
    state_after: str


def text_without_comments(text: str) -> str:
    lines = text.split("\n")
    # Drop empty lines
    lines = [line for line in lines if line.strip()]
    in_comment = False
    for line_idx, line in enumerate(lines):
        new_line = ""
        for idx in range(len(line)):
            if idx < len(line) - 1 and line[idx:idx + 2] == "/-":
                in_comment = True
            if not in_comment:
                new_line += line[idx]
            elif idx > 0 and line[idx - 1: idx + 1] == "-/":
                in_comment = False
        lines[line_idx] = new_line
    # Drop empty lines
    single_removed = [line[:line.index("--")] if "--" in line else line for line in lines]
    single_removed = [line for line in single_removed if line.strip()]
    text_without = "\n".join(single_removed)
    return text_without


def _find_leftmost_not_in_parenthesis(text, substr):
    """
    Get the leftmost instance of a substring that is not in a parenthesis
    :param text: The text to search
    :param substr: The substring to search for
    :return: index of the substring, or -1 if not found
    """
    index = -1
    parenthesis_level = 0
    for i in range(len(text) - len(substr) + 1):
        if text[i] == "(" or text[i] == "[" or text[i] == "{":
            parenthesis_level += 1
        elif text[i] == ")" or text[i] == "]" or text[i] == "}":
            parenthesis_level -= 1
        if text[i:i + len(substr)] == substr and parenthesis_level == 0:
            index = i
            break
    return index


def _find_rightmost_not_in_parenthesis(text, substr):
    """
    Get the rightmost instance of a substring that is not in a parenthesis
    :param text: The text to search
    :param substr: The substring to search for
    :return: index of the substring, or -1 if not found
    """
    index = -1
    parenthesis_level = 0
    for i in range(len(text) - len(substr), -1, -1):
        if text[i] == ")":
            parenthesis_level += 1
        elif text[i] == "(":
            parenthesis_level -= 1
        if text[i:i + len(substr)] == substr and parenthesis_level == 0:
            index = i
            break
    return index


def replace_proof(theorem: str) -> str:
    assert theorem.startswith("theorem ")
    by_matches = BY_REGEX.split(theorem)
    # No by used in the proof
    if len(by_matches) == 1:
        idx = _find_rightmost_not_in_parenthesis(theorem, " :=")
        # Fallback to without space
        if idx == -1:
            idx = _find_rightmost_not_in_parenthesis(theorem, ":=")
        # For pattern matching
        if idx == -1:
            idx = _find_leftmost_not_in_parenthesis(theorem, "|")
        assert idx != -1
        return theorem[:idx] + " := by sorry"

    full_theorem = by_matches[0]
    return full_theorem + " := by sorry"


class Theorem(BaseModel):
    url: str
    commit: str
    file_path: str
    full_name: str
    start: Position
    end: Position
    traced_tactics: list[TracedTactic]

    def to_samples(self) -> Generator[TrainingSample, None, None]:
        # # Ensure correct order of proof states, so that i+1 proof state before == i's proof state after
        # ordered: list[TracedTactic] = []
        # ordered_states = []
        # tac_mask = [False] * len(self.traced_tactics)
        # for idx, tac in enumerate(self.traced_tactics):
        #     if tac.state_after == "no goals":
        #         ordered_states.append(tac.state_before)
        #         ordered.append(tac)
        #         tac_mask[idx] = True
        # while len(ordered_states) < len(self.traced_tactics):
        #     for idx, tac in enumerate(self.traced_tactics):
        #         if tac_mask[idx]:
        #             continue
        #         if tac.state_after in ordered_states:
        #             ordered.insert(ordered_states.index(tac.state_after), tac)
        #             ordered_states.insert(ordered_states.index(tac.state_after), tac.state_before)
        #             tac_mask[idx] = True
        # for i in range(len(ordered) - 1):
        #     assert ordered[i].state_after == ordered[i+1].state_before or ordered[i].state_after == "no goals"
        #
        tactics_so_far = []
        # for tactic in ordered:
        for tactic in self.traced_tactics:
            yield TrainingSample(proof_state=tactic.state_before, tactic=tactic.tactic, tactics_so_far=tactics_so_far)
            tactics_so_far.append(tactic.tactic)

    def to_proof_state(self, handler: LeanREPLHandler, repo_path: Path) -> LeanREPLProofState:
        """Manifest the theorem in the Lean REPL and return the corresponding proof state.

        :param handler: The Lean repl handler to manifest the theorem in
        :param repo_path: Repository path to the root folder of the repo of this theorem
        :return: A proof state
        """
        full_path = repo_path / self.file_path
        handler.send_file(full_path, all_tactics=True)
        response, _ = handler.receive_json()
        # Known bug in Lean REPL
        if response.get("message") == "unknown metavariable '?[anonymous]'":
            raise UnknownMetaVariableError("Unknown metavariable '?[anonymous]'")
        tactics = response["tactics"]
        contains_error = any(msg.severity == "error" for msg in response.get("messages", []))
        if contains_error:
            raise RuntimeError("Error in manifesting theorem")
        for tactic in tactics:
            if tactic["pos"]["line"] >= self.start.line:
                compare_self = tactic["goals"].strip().replace("\n", " ")
                compare_self = re.sub(WS, "", compare_self)
                compare_other = self.traced_tactics[0].state_before.strip().replace("\n", " ")
                compare_other = re.sub(WS, "", compare_other)
                assert compare_self == compare_other

                # it says goals, but since this is the start of the proof state, should only be one
                assert tactic["goals"].count("⊢") == 1
                tactic["goal"] = tactic["goals"]
                return LeanREPLProofState.model_validate(tactic)
        raise RuntimeError("This should never happen!")

    def _lines(self, repo_path) -> list[str]:
        full_path = repo_path / self.file_path
        assert full_path.exists()
        with full_path.open("r") as file:
            lines = file.readlines()
        return lines

    def theorem_statement(self, repo_path: Path) -> str:
        lines = self._lines(repo_path)
        if self.start.line == self.end.line:
            theorem_text = lines[self.start.line - 1][self.start.column - 1:self.end.column]
        else:
            theorem_text = lines[self.start.line - 1][self.start.column - 1:]
            for i in range(self.start.line, self.end.line - 1):
                theorem_text += lines[i]
            theorem_text += lines[self.end.line - 1][:self.end.column]
        text_wo = text_without_comments(theorem_text).strip()
        text_wo = ATTRIBUTE_REGEX.sub("", text_wo).strip()
        text_wo = text_wo.removeprefix("private").strip()
        text_wo = text_wo.removeprefix("protected").strip()
        text_wo = text_wo.removeprefix("nonrec").strip()

        if text_wo.startswith("lemma "):
            text_wo = text_wo.replace("lemma ", "theorem ", 1)
        assert text_wo.startswith("theorem")
        text_wo = replace_proof(text_wo)

        text_before = "\n".join(lines[:self.start.line - 1] + [lines[self.start.line - 1][:self.start.column - 1]])
        text_before_wo = text_without_comments(text_before)
        # Remove imports
        # line_idx = 0
        # for line_idx, line in enumerate(text_before_wo.split("\n")):
        #     if not line.strip().startswith("import"):
        #         break
        # text_before_wo = "\n".join(text_before_wo.split("\n")[line_idx:])

        # imports = self.imports(repo_path)
        handler = LeanREPLHandler(Path("../leanproject"))
        # handler.env = None
        # handler.send_command("\n".join(imports))
        # response, env = handler.receive_json()
        # handler.env = env
        handler.send_command(text_before_wo)
        response, env = handler.receive_json()
        handler.env = env
        handler.send_command(text_wo)
        response, env = handler.receive_json()
        assert len(response["sorries"]) == 1
        assert not any(msg.severity == "error" for msg in response.get("messages", []))
        assert isinstance(response["sorries"][0], LeanREPLProofState) # asserts it worked
        return text_wo

    def imports(self, repo_path: Path) -> list[str]:
        lines = self._lines(repo_path)
        text = "".join(lines)
        text_wo = text_without_comments(text)
        namespace: str | None = None
        opened = {None: []}
        is_in_comment = False
        # Can't use text_wo here because it has different amount of lines
        for idx, line in enumerate(lines):
            if idx >= self.end.line:
                break
            while "/-" in line and "-/" in line:
                line = line.split("/-")[0] + "-/".join(line.split("-/")[1:])
            if "/-" in line and not "-/" in line:
                is_in_comment = True
            elif is_in_comment and "-/" in line:
                is_in_comment = False
                line = line.split("-/")[1]
            if "--" in line:
                line = line[:line.index("--")]
            if is_in_comment:
                continue
            if line.strip().startswith("namespace"):
                old_namespace = namespace
                namespace += line.strip().split(" ")[1]
                opened[namespace] = opened[old_namespace]
            elif line.strip().startswith("end"):
                if namespace is None:
                    continue
                # can also be a section, in which case ignore
                name = line.strip().split(" ")[1]
                if namespace.endswith(name):
                    del opened[namespace]
                namespace = namespace.removesuffix(name)
            elif line.strip().startswith("open"):
                open_str = " ".join(line.strip().split(" ")[1:]).strip().removeprefix("scoped ").strip().removesuffix(
                    " in")
                opened[namespace].extend(open_str)

        imports = [line for line in text_wo.split("\n") if line.strip().startswith("import")]
        return imports


def parse_json(json_path: Path) -> Generator[Theorem, None, None]:
    with json_path.open("r") as f:
        data = json.load(f)
    for entry in data:
        # Happens exactly 5 times on the full dataset
        if entry["full_name"] is None:
            continue
        yield Theorem.model_validate(entry)


class TheoremDataset(Dataset):
    def __init__(self, json_path: Path):
        self.json_path = json_path
        self.thms = list(parse_json(json_path))

    def __len__(self):
        return len(self.thms)

    def __getitem__(self, item):
        return self.thms[item]


class TrainSampleDataset(TheoremDataset):
    def __init__(self, json_path: Path):
        super().__init__(json_path)
        self.samples = [sample for thm in self.thms for sample in thm.to_samples()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item) -> TrainingSample:
        return self.samples[item]


if __name__ == '__main__':
    from lean_repl_py import LeanREPLHandler

    handler = LeanREPLHandler(Path("../leanproject"))
    train_data = TheoremDataset(LEAN_DOJO_PATH / "train.json")
    valid_data = TheoremDataset(LEAN_DOJO_PATH / "val.json")
    test_data = TheoremDataset(LEAN_DOJO_PATH / "test.json")
    for thm in parse_json(LEAN_DOJO_PATH / "train.json"):
        # statement = thm.theorem_statement(Path("../mathlib4"))
        # imports = thm.imports(Path("../mathlib4"))
        # handler.send_command("\n".join(imports))
        # response, env = handler.receive_json()
        # handler.env = env
        thm.to_proof_state(handler, Path("../mathlib4"))
        # handler.env = None
        # handler.send_command("\n".join(imports))
        # response, env = handler.receive_json()
        # print(response)
        # handler.env = env
        # handler.send_command(statement)
        # response, env = handler.receive_json()
        # print(response)
