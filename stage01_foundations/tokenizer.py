# stage01_foundations/tokenizer.py

import torch

class CharacterTokenizer:
    """
    Character-level tokenizer.

    We intentionally keep this simple so that every step of the pipeline is visible.

    Design decisions:

    CASE:
        By default, we preserve case → 'H' and 'h' are treated as different tokens.
        If needed, we can normalize using lowercase=True, but this must remain
        consistent once training starts.

    UNKNOWN CHARACTERS:
        Any unseen character is mapped to <UNK> instead of raising an error.
        This avoids failures during inference when new inputs appear.

    PADDING:
        We right-pad sequences using PAD_ID=0.
        Padding is not ignored because embeddings are zero — it is ignored because
        we explicitly pass an attention mask to the model.

    SPECIAL TOKENS:
        PAD=0, SOS=1, EOS=2, UNK=3
        These IDs are fixed. Changing them later will break saved models.
    """

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase

        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

        # Special tokens (fixed IDs)
        self.PAD_TOKEN = "<PAD>"; self.PAD_ID = 0
        self.SOS_TOKEN = "<SOS>"; self.SOS_ID = 1
        self.EOS_TOKEN = "<EOS>"; self.EOS_ID = 2
        self.UNK_TOKEN = "<UNK>"; self.UNK_ID = 3

        # Register special tokens first
        self._add_token(self.PAD_TOKEN)
        self._add_token(self.SOS_TOKEN)
        self._add_token(self.EOS_TOKEN)
        self._add_token(self.UNK_TOKEN)


    def _add_token(self, token: str) -> int:
        """Adding token to vocab if not present."""
        if token not in self.char_to_id:
            new_id = len(self.char_to_id)
            self.char_to_id[token] = new_id
            self.id_to_char[new_id] = token
        return self.char_to_id[token]

    def _normalize(self, text: str) -> str:
        """Apply normalization consistently."""
        return text.lower() if self.lowercase else text

    # -----------------------------------------------------------------------

    def build_vocab(self, corpus: list[str]) -> None:
        """
        Scan corpus and register every unique character.

        WHY pass all splits? If validation text contains 'z' but training
        does not, 'z' hits UNK at eval time. Pass train + val + test combined
        so vocab is complete. UNK still handles truly unseen inference chars.

        Args:
            corpus: All raw strings across every data split.
        """
        
        all_chars = set()
        for text in corpus:
            all_chars.update(self._normalize(text))

        # Sorting ensures deterministic ID assignment
        for char in sorted(all_chars):
            self._add_token(char)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Special tokens: PAD={self.PAD_ID}, SOS={self.SOS_ID}, EOS={self.EOS_ID}, UNK={self.UNK_ID}")

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    # -----------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_sos: bool = False,
        add_eos: bool = False,
        max_len: int | None = None,
    ) -> list[int]:
        """
        Convert string → list of token IDs.

        Important detail:
        We truncate content BEFORE adding special tokens.
        This guarantees that SOS/EOS are never accidentally removed.
        """
        text = self._normalize(text)

        content = [self.char_to_id.get(c, self.UNK_ID) for c in text]

        if max_len is not None:
            reserved = (1 if add_sos else 0) + (1 if add_eos else 0)
            content = content[:max(0, max_len - reserved)]

        ids = []
        if add_sos:
            ids.append(self.SOS_ID)

        ids.extend(content)

        if add_eos:
            ids.append(self.EOS_ID)

        return ids

    # -----------------------------------------------------------------------

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convert token IDs -> string."""
        special_ids = {self.PAD_ID, self.SOS_ID, self.EOS_ID, self.UNK_ID}
        chars = []

        for id_ in ids:
            if id_ not in self.id_to_char:
                raise KeyError(f"ID {id_} not in vocabulary")

            if skip_special and id_ in special_ids:
                continue

            chars.append(self.id_to_char[id_])

        return "".join(chars)

    # -----------------------------------------------------------------------

    def encode_batch(
        self,
        texts: list[str],
        add_sos: bool = False,
        add_eos: bool = False,
        max_len: int | None = None,
        return_tensors: bool = False,
    ):
        """
        Encode a batch of strings, pad to equal length, return attention mask.

        WHY RETURN MASK HERE?
          The tokenizer is the only place that knows which positions are real
          and which are padding. If we compute the mask later, we'd have to
          re-derive it from the token IDs (checking == PAD_ID everywhere).
          It's cleaner and safer to produce it once at the source.

        WHY IS THE MASK THE CONTRACT, NOT THE ZERO EMBEDDING?
          PAD tokens have near-zero embeddings (padding_idx=0 in nn.Embedding
          freezes them). But "near-zero" is not "exactly zero" after scaling,
          and it says nothing about the attention score. Without the mask,
          attention still computes scores for PAD positions and attends to them.
          The mask explicitly sets those attention logits to -inf before softmax,
          making them exactly zero in the attention weights. That is the real fix.

        Args:
            texts:          List of raw strings.
            add_sos:        Prepend SOS to each sequence.
            add_eos:        Append EOS to each sequence.
            max_len:        Truncate content to this total length (specials preserved).
            return_tensors: If True, return torch.Tensors instead of plain lists.

        Returns:
            padded  : [B, T_max]  token IDs, right-padded with PAD_ID
            mask    : [B, T_max]  1 = real token, 0 = PAD (attention ignore)
            lengths : [B]         true length of each sequence before padding

        Shape contract (for everything downstream):
            padded  → dtype=torch.long,  used by nn.Embedding
            mask    → dtype=torch.bool,  used by attention layers
            lengths → dtype=torch.long,  used by masking utilities
        Mask:
            1 → valid token
            0 → padding (should be ignored by attention)
        """
        encoded = [
            self.encode(t, add_sos=add_sos, add_eos=add_eos, max_len=max_len)
            for t in texts]

        lengths = [len(seq) for seq in encoded]
        T_max = max(lengths)

        padded = [seq + [self.PAD_ID] * (T_max - len(seq)) for seq in encoded]

        mask = [
            [1] * length + [0] * (T_max - length)
            for length in lengths]

        if return_tensors:
            return (
                torch.tensor(padded, dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool),
                torch.tensor(lengths, dtype=torch.long),
            )

        return padded, mask, lengths


# Driver Code
if __name__ == "__main__":

    tokenizer = CharacterTokenizer()
    corpus = ["Hello World", "I am learning Transformer", "Hey Hi"]
    tokenizer.build_vocab(corpus)

    text = "Hello"
    ids = tokenizer.encode(text, add_sos=True, add_eos=True)
    decoded = tokenizer.decode(ids)

    print("Original:", text)
    print("Encoded :", ids)
    print("Decoded :", decoded)

    batch = ["Hi", "Hello", "Hey"]
    padded, mask, lengths = tokenizer.encode_batch(batch)

    print("\nBatch:")
    print("Padded :", padded)
    print("Mask   :", mask)
    print("Lengths:", lengths)