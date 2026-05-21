"""Atom-level SMILES tokenizer.

Splits SMILES strings into chemically meaningful atom-level tokens.
Used as the base tokenization for SPE (SMILES Pair Encoding).

Handles:
- Bracket atoms: [nH], [O-], [C@@H], [NH4+], etc.
- Two-letter elements: Br, Cl, Si, Na, Li, Mg, Ca, Al
- Stereochemistry: @@, @
- Bond symbols: -, =, #, :, /, \\ .
- Ring closures: % followed by 2 digits, or single digits
- Branch parentheses: (, )
- Aromatic atoms (lowercase): c, n, o, s, p, b
- Aliphatic atoms (uppercase single-letter): C, N, O, S, P, F, I, B, H
"""

import re
from typing import List

# Ordered by priority: longer/more specific patterns first
_ATOM_PATTERNS = [
    # Bracket atoms with internal content: [nH], [O-], [C@@H], [NH4+], [235U], etc.
    (r"\[[^\]]*\]", "bracket"),
    # Two-letter element symbols
    (r"Br|Cl|Si|Na|Li|Mg|Ca|Al", "element2"),
    # Stereochemistry markers
    (r"@@|@", "stereo"),
    # Bond symbols
    (r"/|\\|\.|=", "bond"),
    # Ring closures with % prefix (multi-digit)
    (r"%\d{2}", "ring_pct"),
    # Single-digit ring closures
    (r"\d", "ring"),
    # Aromatic bond ':'
    (r":", "aromatic_bond"),
    # Branch parentheses
    (r"\(|\)", "branch"),
    # Triple bond '#'
    (r"#", "triple_bond"),
    # Aromatic atoms (lowercase, after element2 to avoid partial match)
    (r"[cnospb]", "aromatic"),
    # Aliphatic single-letter atoms (uppercase)
    (r"[CNOSPFIBH]", "element1"),
    # Catch-all for any remaining non-whitespace
    (r"[^\[\]\(\)\s]", "other"),
]

# Deduplicated combined regex
_ATOM_RE = re.compile("|".join(f"(?:{p})" for p, _ in _ATOM_PATTERNS))


def atom_tokenize(smiles: str) -> List[str]:
    """Split a SMILES string into atom-level tokens.

    Args:
        smiles: A SMILES string.

    Returns:
        List of atom-level token strings.

    Examples:
        >>> atom_tokenize("CCO")
        ['C', 'C', 'O']
        >>> atom_tokenize("c1ccccc1")
        ['c', '1', 'c', 'c', 'c', 'c', 'c', '1']
        >>> atom_tokenize("[NH4+]")
        ['[NH4+]']
        >>> atom_tokenize("C[C@H](O)Cl")
        ['C', '[C@H]', '(', 'O', ')', 'Cl']
        >>> atom_tokenize("C/C=C\\C")
        ['C', '/', 'C', '=', 'C', '\\', 'C']
    """
    s = smiles.strip()
    tokens = []
    pos = 0
    while pos < len(s):
        matched = False
        for pattern, _ in _ATOM_PATTERNS:
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


def atom_detokenize(tokens: List[str]) -> str:
    """Join atom-level tokens back into a SMILES string.

    Args:
        tokens: List of atom-level token strings.

    Returns:
        SMILES string (concatenation of tokens).
    """
    return "".join(tokens)
