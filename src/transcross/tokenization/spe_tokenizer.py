"""SMILES Pair Encoding (SPE) tokenizer.

Implements byte-pair encoding style vocabulary learning on top of
atom-level SMILES tokenization. Learns merge rules from training
SMILES only, then applies them greedily at tokenization time.
"""

import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .atom_tokenizer import atom_tokenize, atom_detokenize

# Special tokens with fixed IDs
_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}
_NUM_SPECIAL = len(_SPECIAL_TOKENS)


def _pair_counts(token_lists: List[List[str]]) -> Counter:
    """Count adjacent token pair frequencies across all sequences."""
    counts: Counter = Counter()
    for tokens in token_lists:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            counts[pair] += 1
    return counts


def _apply_merge(tokens: List[str], pair: Tuple[str, str]) -> List[str]:
    """Apply a single merge rule greedily to a token list."""
    new_tokens: List[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(pair[0] + pair[1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


def _apply_merges(tokens: List[str], merges: List[Tuple[str, str]]) -> List[str]:
    """Apply all merge rules in order to a token list."""
    for pair in merges:
        tokens = _apply_merge(tokens, pair)
    return tokens


class SPETokenizer:
    """SMILES Pair Encoding tokenizer.

    Learns a subword vocabulary from atom-level SMILES tokens by
    iteratively merging the most frequent adjacent token pairs.

    Attributes:
        pad_id: Padding token ID (always 0).
        bos_id: Beginning-of-sequence token ID (always 1).
        eos_id: End-of-sequence token ID (always 2).
        unk_id: Unknown token ID (always 3).
        vocab_size: Total vocabulary size (including special tokens).
    """

    def __init__(self):
        self._token_to_id: Dict[str, int] = dict(_SPECIAL_TOKENS)
        self._id_to_token: Dict[int, str] = {v: k for k, v in _SPECIAL_TOKENS.items()}
        self._merges: List[Tuple[str, str]] = []
        self._merge_ordering: List[int] = []  # track which merge created each vocab entry

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    @property
    def pad_id(self) -> int:
        return _SPECIAL_TOKENS["<pad>"]

    @property
    def bos_id(self) -> int:
        return _SPECIAL_TOKENS["<bos>"]

    @property
    def eos_id(self) -> int:
        return _SPECIAL_TOKENS["<eos>"]

    @property
    def unk_id(self) -> int:
        return _SPECIAL_TOKENS["<unk>"]

    @property
    def num_merges(self) -> int:
        return len(self._merges)

    def train(
        self,
        smiles_list: List[str],
        vocab_size: int = 256,
        min_frequency: int = 2,
    ) -> int:
        """Train SPE vocabulary on a list of SMILES strings.

        Args:
            smiles_list: Training SMILES strings.
            vocab_size: Target vocabulary size (including special tokens).
            min_frequency: Minimum pair frequency for a merge to be accepted.

        Returns:
            Number of merge operations performed.
        """
        # Start from atom-level tokenization for all training SMILES
        token_lists = [atom_tokenize(s) for s in smiles_list]

        # Collect all unique atom-level tokens from training set
        atom_tokens: set = set()
        for tokens in token_lists:
            atom_tokens.update(tokens)

        # Add all observed atom-level tokens to vocabulary
        for tok in sorted(atom_tokens):
            if tok not in self._token_to_id:
                self._token_to_id[tok] = len(self._token_to_id)

        self._merges = []

        # Iterative merging
        current_lists = token_lists
        while len(self._token_to_id) < vocab_size:
            counts = _pair_counts(current_lists)
            if not counts:
                break

            best_pair, best_count = counts.most_common(1)[0]
            if best_count < min_frequency:
                break

            # Record merge
            self._merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]

            # Add to vocabulary
            if merged_token not in self._token_to_id:
                self._token_to_id[merged_token] = len(self._token_to_id)

            # Apply merge to all sequences
            current_lists = [_apply_merge(tokens, best_pair) for tokens in current_lists]

        # Rebuild id_to_token
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}
        return len(self._merges)

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a SMILES string into SPE tokens.

        Args:
            smiles: Input SMILES string.

        Returns:
            List of SPE token strings (no special tokens added).
        """
        tokens = atom_tokenize(smiles)
        tokens = _apply_merges(tokens, self._merges)
        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """Join SPE tokens back into a SMILES string.

        Args:
            tokens: List of SPE token strings.

        Returns:
            SMILES string.
        """
        return "".join(tokens)

    def encode(
        self,
        smiles: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """Encode a SMILES string to SPE token IDs.

        Args:
            smiles: Input SMILES string.
            add_bos: Whether to prepend BOS token.
            add_eos: Whether to append EOS token.

        Returns:
            List of token IDs.
        """
        tokens = self.tokenize(smiles)
        ids = [self._token_to_id.get(t, self.unk_id) for t in tokens]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], remove_special: bool = True) -> str:
        """Decode token IDs back to a SMILES string.

        Args:
            ids: List of token IDs.
            remove_special: If True, strip pad, bos, eos, unk tokens.

        Returns:
            SMILES string.
        """
        special = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        tokens = []
        for tid in ids:
            tok = self._id_to_token.get(tid, "<unk>")
            if remove_special and tid in special:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def get_merge_rules(self) -> List[Dict]:
        """Return merge rules as a list of {pair, merged} dicts."""
        return [
            {"pair": list(p), "merged": p[0] + p[1]}
            for p in self._merges
        ]

    def save(self, path: str) -> None:
        """Save vocabulary and merge rules to a JSON file."""
        data = {
            "token_to_id": self._token_to_id,
            "merges": [list(p) for p in self._merges],
            "vocab_size": len(self._token_to_id),
            "num_merges": len(self._merges),
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SPETokenizer":
        """Load vocabulary and merge rules from a JSON file."""
        tokenizer = cls()
        with open(path) as f:
            data = json.load(f)
        tokenizer._token_to_id = data["token_to_id"]
        tokenizer._id_to_token = {
            int(v): k for k, v in data["token_to_id"].items()
        }
        tokenizer._merges = [tuple(p) for p in data.get("merges", [])]
        return tokenizer
