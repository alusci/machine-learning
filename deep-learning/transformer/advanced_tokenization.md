## BPE (Byte Pair Encoding)

Split words into subwords:

- For example, playfulness is split into "play", "ful", "ness"
- Overall, BPE starts with character level tokens and merges the most frequent adjiacent pairs until a vocabulary is built
- This helps with OOV (out of vocab) words and keeps the vocabulary within 30k-50k tokens. 
- Models: GPT, BERT

## Sentence Piece (BPE Variant)

Split words into subwords:

- For example, I dislike pizza is split into "_I", "_dis", "like", "_pizz"
- Notice how a special character (_) indicates the presence of a white space 
- For this reason, Sentence Piece can be directly applied to raw text
- BPE requires to split text on white spaces
- This aspect makes Sentece Piece model agnostic
- Models: T5



                                                                                                                                               


