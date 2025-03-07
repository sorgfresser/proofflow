"""
Code to create a BPE tokenizer for Lean 4 on the mathlib data used in lean dojo.
"""

from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import PreTokenizer, Whitespace, Punctuation, Sequence
from pathlib import Path

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
vocab_size = 50_257
trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[EOS]", "[PROOFSTATE]", "[PROOFSTEP]", "[TACTICS]", "[SEP]"])

# We split on whitespaces or periods, as periods are used frequently in imports
# For example, we do not want multiple namespaces in one token
# tokenizer.pre_tokenizer = Split(Regex(r"\w+|[^\w\s]+|\."), behavior="isolated")
# tokenizer.pre_tokenizer = Whitespace()
tokenizer.pre_tokenizer = Sequence([Punctuation(), Whitespace()])
mathlib_path = Path("../mathlib4")
files = [str(file.resolve()) for file in mathlib_path.glob("**/*.lean")]

tokenizer.train(files, trainer)
tokenizer.save("lean_tokenizer.json")
