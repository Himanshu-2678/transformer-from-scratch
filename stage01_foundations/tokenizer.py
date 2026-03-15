class CharacterTokenizer:
    """
    We are using Character-level Tokenizer. Its the simplest possible Tokenizer - no external library needed.
    Each char in the training corpus becomes a vector.

    Vocabulary is built from the dataset itself - no pre-defined vocab list.
    
    """

    def __init__(self):
        # self.char_to_id maps each char -> its integer ID
        # e.g. {"a": 0, "b": 1, ....}
        self.char_to_id: dict[str, int] = {}

        # self.id_to_char maps each integer ID -> its character 
        self.id_to_char: dict[int, str] = {}

        # reserve ID 0 for special <PAD> token
        # PAD is used to fill sequences of equal length
        # Without this, PyTorch cannot stack variable length sequences into tensors
        self.PAD_TOKEN = "<PAD>"
        self.PAD_ID = 0

        # <SOS>: Start of Sequence. Prepend to every decoder input
        # its like saying Decoder to start generating now.
        self.SOS_TOKEN = "<SOS>"
        self.SOS_ID = 1

        # <EOS>: End of Sequence. Append to every target sequence.
        # it is for stopping the Decoder to generate.
        self.EOS_TOKEN = "<EOS>"
        self.EOS_ID = 2

        # register the special tokens first so that they always have IDs: 0, 1, 2, etc.
        # PAD ID must be consistent accross entire codebase (this is what we will use for masking later.)
        self._add_token(self.PAD_TOKEN)
        self._add_token(self.SOS_TOKEN)
        self._add_token(self.EOS_TOKEN)

    def _add_token(self, token: str) -> int:
        """
        We will add a single token to the vocab if not already present.
        Return the tokens ID.
        """
        if token not in self.char_to_id:
            new_id = len(self.char_to_id)
            self.char_to_id[token] = new_id
            self.id_to_char[new_id] = token
        
        return self.char_to_id[token]
    
    def build_vocab(self, corpus: list[str]) -> None:
        """
        Scan a list of string and add every unique character to vocab.

        Args: corpus = list of str (training + validation data)
        """

        # collect all the unique characters from entire corpus
        all_chars = set()
        for text in corpus:
            all_chars.update(set(text))
        
        # same corpus always produces same vocab
        # Without sorting, set iteration order is arbitrary and our
        # token IDs would change between runs. That breaks saved models.
        for char in sorted(all_chars):
            self._add_token(char)

        print(f"Vocabulary built: {self.vocab_size} tokens")
        print(f" Special tokens: PAD={self.PAD_ID}, SOS={self.SOS_ID}, EOS={self.EOS_ID}")
        print(f" Sample chars: {list(self.char_to_id.items())[3:8]}")

    @property
    def vocab_size(self) -> int:
        """
        Total Vocabulary size: V
        This is a @property so it always reflects the current state of the vocab
        We will pass this number into our Embedding layer as nn.Embedding(V, d_model).
        """
        return len(self.char_to_id)
    
    def encode(self, text: str, add_sos: bool = False, add_eos: bool=False) -> list[int]:
        """
        Convert the string to list of integer ID.
        Args:
            text: raw string to encode
            add_sos: prepend <SOS> token. Used for decoder input sequence
            add_eos: append <EOS> token. used for target input sequence so the model learns when to stop

        returns: List of integers IDs = [1, 3, 7, 4, ...] for <SOS>hello<EOS>
        Shape: [T] where T = len(text) + (1 if add_sos) + (1 if add_eos)
        """

        ids = []

        if add_sos:
            ids.append(self.SOS_ID)

        for char in text:
            if char not in self.char_to_id:
                # this should not happen after build_vocab()
                raise KeyError(
                    f"Character '{char}' not in vocabulary."
                    f"Did you call build_vocab() with all your data?"
                )
            ids.append(self.char_to_id[char])

        if add_eos:
            ids.append(self.EOS_ID)
        
        return ids
    
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        Covert the list of Integer ID back to string.
        Args:
            ids: list integer token IDs
            skip_special: If True, skip PAD, SOS, EOS tokens in output.
                          Set False when debugging to see raw token string
        
        returns: Decoded String
        """
        special_ids = {self.PAD_ID, self.SOS_ID, self.EOS_ID}
        chars = []

        for id_ in ids:
            if id_ not in self.id_to_char:
                raise KeyError(f"ID '{id_}' not in vocab.")

            if skip_special and id_ in special_ids:
                continue

            chars.append(self.id_to_char[id_])

        return "".join(chars)  

    def encode_batch(
            self, 
            texts: list[str],
            add_sos: bool = False,
            add_eos: bool = False,
            max_len: int | None = None,
    ) -> tuple[list[list[int]], list[int]]:
        """
        Encode a batch of strings and PAD them to equal lengths.
        Why PAD? PyTorch's nn.Embedding and all subsequent operations expect batched tensors
        of shape [Batch, SeqLen]. Every sequence in a batch must have same length. 
        We PAD shorted sequences with PAD_ID(0) on right.

        Args:
            texts:   List of strings to encode
            add_sos: Prepend SOS to each sequence
            add_eos: Append EOS to each sequence
            max_len: If set, truncate or use as padding target.
                     If None, pad to the longest sequence in the batch.

        Returns:
            Tuple of:
                - padded_ids: list[list[int]] — each inner list has the same length
                - lengths:    list[int]       — actual (pre-padding) length of each seq

        Shape: ([B, T_max], [B])
        """

        # step-1: encode every string sequentially 
        encoded = [self.encode(t, add_sos=add_sos, add_eos=add_eos) for t in texts]

        # step-2: taking true lengths before padding (needed for masking later)
        lengths = [len(ids) for ids in encoded]

        # step-3: determine target padded length
        T_max = max_len if max_len is not None else max(lengths)

        # Step 4: Pad each sequence on the RIGHT with PAD_ID
        # Right-padding is standard. The model will learn to ignore PAD positions via an attention mask.
        padded = []
        for ids in encoded:
            if len(ids) > T_max:
                padded.append(ids[:T_max])         # truncating if too long
            else:
                pad_count = T_max - len(ids)
                padded.append(ids + [self.PAD_ID] * pad_count)

        return padded, lengths

# Driver Code 
if __name__ == "__main__":
    tokenizer = CharacterTokenizer()
    corpus = ["Hello World", "I am learning Transformer", "Hey Hi"]
    tokenizer.build_vocab(corpus)

    # encoding and decoding
    text = "Hello"
    ids = tokenizer.encode(text, add_sos=True, add_eos=True)
    decoded = tokenizer.decode(ids)

    print(f"Original:  '{text}'")
    print(f"Encoded:   {ids}")       # e.g. [1, 10, 7, 14, 14, 15, 2]
    print(f"Decoded:   '{decoded}'") # 'Hello'
    assert decoded == text, "Round-trip failed!"

    # Test batch encoding with padding
    batch = ["Hi", "Hello", "Hey"]
    padded, lengths = tokenizer.encode_batch(batch)
    print(f"Batch padded: {padded}")  # all same length, short ones have trailing 0s
    print(f"True lengths: {lengths}")


"""
If a character appears in inference time which is not present in vocabulary, it will throw an error.
This is called out-of-vocabulary(OOV) problem. 
Therefore we added <UNK> token. If a character appear which is not in vocabulary, instead of crashing,
it silently maps it to <UNK> and continue.
"""