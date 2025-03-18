"""
Code to create a BPE tokenizer for Lean 4 on the mathlib data used in lean dojo.
"""

from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from proofflow.data import TheoremDataset, LEAN_DOJO_PATH, TrainSampleDataset
from tokenizers.pre_tokenizers import PreTokenizer, WhitespaceSplit, Punctuation, Sequence
from pathlib import Path
from transformers import PreTrainedTokenizerFast


def iterate_samples(datasets: list[TrainSampleDataset]):
    for dataset in datasets:
        for idx in range(len(dataset)):
            yield "\n".join([dataset[idx].tactic, dataset[idx].proof_state] + dataset[idx].tactics_so_far + dataset[
                idx].proof_states_so_far)


if __name__ == '__main__':
    train_data = TrainSampleDataset(LEAN_DOJO_PATH / "train.json")
    valid_data = TrainSampleDataset(LEAN_DOJO_PATH / "val.json")
    test_data = TrainSampleDataset(LEAN_DOJO_PATH / "test.json")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    vocab_size = 50_257
    trainer = BpeTrainer(vocab_size=vocab_size,
                         special_tokens=["[UNK]", "[PAD]", "[EOS]", "[PROOFSTATE]", "[PROOFSTEP]", "[TACTICS]",
                                         "[SEP]", "[STATESEP]", "[SUC]", "[INC]", "[INV]"])

    # We split on whitespaces or periods, as periods are used frequently in imports
    # For example, we do not want multiple namespaces in one token
    # tokenizer.pre_tokenizer = Split(Regex(r"\w+|[^\w\s]+|\."), behavior="isolated")
    tokenizer.pre_tokenizer = WhitespaceSplit()
    # tokenizer.pre_tokenizer = Sequence([Punctuation(), Whitespace()])
    # mathlib_path = Path("../mathlib4")
    # files = [str(file.resolve()) for file in mathlib_path.glob("**/*.lean")]
    #
    # tokenizer.train(files, trainer)
    # tokenizer.save("lean_tokenizer.json")
    tokenizer.train_from_iterator(iterate_samples([train_data, valid_data, test_data]), trainer)
    #tokenizer.save("lean_tokenizer.json")


    transformers_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    transformers_tokenizer.add_special_tokens({"eos_token": "[EOS]", "pad_token": "[PAD]", "unk_token": "[UNK]", "sep_token": "[SEP]"})
    transformers_tokenizer.save_pretrained("lean_tokenizer2")

