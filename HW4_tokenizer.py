from tokenizers import Tokenizer, pre_tokenizers, normalizers, trainers, models
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.trainers import BpeTrainer
from collections import defaultdict

# Sample training data for the tokenizers
training_data = [
    "he is fair and aware and responsive",
    "he is unfair and unaware and unresponsive",
    "he is unhappy and unkind and unpleasant",
    "is he fair and aware and responsive",
    "is he unhappy and unkind and unaware"
]

# Sentence to encode
sentence_to_encode = "he is unfair and unaware and unresponsive"

# Define tokenizer training function
def train_and_encode_bpe(min_frequency, vocab_size, text_data, sentence):
    # Initialize a tokenizer using the BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Pre-tokenization with whitespace splitting
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Normalization
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

    # Trainer with the specified vocab_size and min_frequency
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
    
    # Train the tokenizer
    tokenizer.train_from_iterator(text_data, trainer)
    
    # Encode the given sentence
    encoding = tokenizer.encode(sentence)
    
    return encoding.tokens

# Train and encode using the different hyper-parameter configurations
results = defaultdict(list)
hyperparams = [
    {"min_frequency": 2, "vocab_size": 20},
    {"min_frequency": 2, "vocab_size": 50},
    {"min_frequency": 3, "vocab_size": 50}
]

# Train tokenizers and encode the sentence with each configuration
for params in hyperparams:
    tokens = train_and_encode_bpe(params["min_frequency"], params["vocab_size"], training_data, sentence_to_encode)
    results[f'min_freq={params["min_frequency"]}, vocab_size={params["vocab_size"]}'] = tokens

# Output the results
for key, tokens in results.items():
    print(f"{key}: {tokens}")
