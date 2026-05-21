"""Regex-based SMILES tokenizer with special token support."""

import json
import re
from typing import Dict, List, Optional


# Tokenization patterns ordered by priority (longer matches first)
_SMILES_PATTERNS = [
    # Bracket atoms: [nH], [O-], [Na+], [CH3], [235U], etc.
    (r"\[[^\]]*\]", "bracket"),
    # Two-letter elements: Br, Cl, Si, Na, Li, Mg, Ca, Al
    (r"Br|Cl|Si|Na|Li|Mg|Ca|Al", "element2"),
    # Stereochemistry: @@, @
    (r"@@|@", "stereo"),
    # Bond symbols: /, \, ., =
    (r"/|\\|\.|=", "bond"),
    # Ring closures: % followed by 2 digits, or single digits
    (r"%\d{2}", "ring_pct"),
    (r"\d", "ring"),
    # Branch parentheses
    (r"\(|\)", "branch"),
    # Aromatic atoms (lowercase): c, n, o, s, p, b (must come after element2)
    (r"[cnospb]", "aromatic"),
    # Aliphatic atoms (uppercase single-letter): C, N, O, S, P, F, I, B, H
    (r"[CNOSPFIBH]", "element1"),
    # Other characters (catch-all, should not normally match)
    (r"[^\[\]\(\)\s]", "other"),
]

_SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


class SmilesTokenizer:
    """Regex-based SMILES tokenizer with vocabulary built from a SMILES corpus."""

    def __init__(self):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._vocab_size: int = 0

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

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

    @staticmethod
    def tokenize_smiles(smiles: str) -> List[str]:
        """Split a SMILES string into tokens using regex patterns."""
        tokens = []
        pos = 0
        s = smiles.strip()
        while pos < len(s):
            matched = False
            for pattern, _ in _SMILES_PATTERNS:
                m = re.match(pattern, s[pos:])
                if m:
                    tokens.append(m.group())
                    pos += len(m.group())
                    matched = True
                    break
            if not matched:
                # Unknown character — treat as single token
                tokens.append(s[pos])
                pos += 1
        return tokens

    @classmethod
    def build_from_smiles(cls, smiles_list: List[str]) -> "SmilesTokenizer":
        """Build vocabulary from a list of SMILES strings."""
        tokenizer = cls()

        token_counts: Dict[str, int] = {}
        for smi in smiles_list:
            for tok in cls.tokenize_smiles(smi):
                token_counts[tok] = token_counts.get(tok, 0) + 1

        # Sort by frequency (descending), then alphabetically
        sorted_tokens = sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))

        # Build vocabulary starting after special tokens
        tokenizer._token_to_id = dict(_SPECIAL_TOKENS)
        for tok, _ in sorted_tokens:
            if tok not in tokenizer._token_to_id:
                tokenizer._token_to_id[tok] = len(tokenizer._token_to_id)

        tokenizer._id_to_token = {v: k for k, v in tokenizer._token_to_id.items()}
        tokenizer._vocab_size = len(tokenizer._token_to_id)
        return tokenizer

    def encode(self, smiles: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode a SMILES string to token IDs."""
        tokens = self.tokenize_smiles(smiles)
        ids = [self._token_to_id.get(t, self.unk_id) for t in tokens]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], remove_special: bool = True) -> str:
        """Decode token IDs back to a SMILES string."""
        special = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        tokens = []
        for tid in ids:
            tok = self._id_to_token.get(tid, "<unk>")
            if remove_special and tid in special:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def save(self, path: str) -> None:
        """Save vocabulary to JSON file."""
        with open(path, "w") as f:
            json.dump({"token_to_id": self._token_to_id}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SmilesTokenizer":
        """Load vocabulary from JSON file."""
        tokenizer = cls()
        with open(path) as f:
            data = json.load(f)
        tokenizer._token_to_id = data["token_to_id"]
        tokenizer._id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        tokenizer._vocab_size = len(tokenizer._token_to_id)
        return tokenizer
