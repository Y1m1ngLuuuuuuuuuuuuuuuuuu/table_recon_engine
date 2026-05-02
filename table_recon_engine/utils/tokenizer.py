import json
import re
from pathlib import Path
from typing import Iterable


CELL_OPEN_RE = re.compile(r"^<td(\s+[^>]*)?>(</td>)?$")


class HTMLTokenizer:
    pad_token = "<pad>"
    sos_token = "<sos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self, tokens: Iterable[str] | None = None) -> None:
        base_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self.token_to_id = {token: idx for idx, token in enumerate(base_tokens)}
        self.id_to_token = list(base_tokens)
        if tokens:
            self.add_tokens(tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[self.sos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def add_tokens(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.id_to_token)
                self.id_to_token.append(token)

    def encode(self, tokens: Iterable[str], add_special: bool = True) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        if add_special:
            return [self.sos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> list[str]:
        special = {self.pad_id, self.sos_id, self.eos_id} if skip_special else set()
        tokens = []
        for idx in ids:
            if idx in special:
                continue
            if 0 <= int(idx) < len(self.id_to_token):
                tokens.append(self.id_to_token[int(idx)])
            else:
                tokens.append(self.unk_token)
        return tokens

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.id_to_token, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HTMLTokenizer":
        id_to_token = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        tokenizer.token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
        tokenizer.id_to_token = list(id_to_token)
        return tokenizer

    @staticmethod
    def is_cell_token(token: str) -> bool:
        return bool(CELL_OPEN_RE.match(token))


def merge_pubtabnet_tokens(tokens: list[str]) -> list[str]:
    """Merge PubTabNet split td tokens into complete opening-cell tags.

    PubTabNet often stores tags as pieces, e.g. ["<td", " colspan=\"2\"", ">"].
    This helper turns that into ["<td colspan=\"2\">"] so the decoder has one
    token per logical cell opening, matching the bbox supervision.
    """

    merged: list[str] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("<td") and not token.endswith(">"):
            parts = [token]
            idx += 1
            while idx < len(tokens):
                parts.append(tokens[idx])
                if tokens[idx].endswith(">"):
                    break
                idx += 1
            merged.append("".join(parts))
        else:
            merged.append(token)
        idx += 1
    return merged


def default_html_tokens() -> list[str]:
    return [
        "<table>",
        "</table>",
        "<thead>",
        "</thead>",
        "<tbody>",
        "</tbody>",
        "<tr>",
        "</tr>",
        "<td>",
        "</td>",
        "<td colspan=\"2\">",
        "<td colspan=\"3\">",
        "<td colspan=\"4\">",
        "<td rowspan=\"2\">",
        "<td rowspan=\"3\">",
        "<td rowspan=\"4\">",
        "<td colspan=\"2\" rowspan=\"2\">",
        "<td rowspan=\"2\" colspan=\"2\">",
    ]
