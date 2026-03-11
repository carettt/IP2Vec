# IP2Vec - Word2Vec-style continuous embedding model

Embedding neural network which embeds IP flow entites (consisting of `src_ip`, `dst_port`, and `protocol`)
to variable dimension vector embeddings. Contains training binary and inference binary with PCA
dimensionality reduction for batches of samples and single flows. Configurable through the CLI or
through the library APIs.

### Dependencies
Both binaries depend on `libtorch` (C++ library for PyTorch). This is not bundled
with the binaries. Ensure `libtorch` is downloaded and is included in the
`LD_LIBRARY_PATH` environment variable:

```zsh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libtorch/lib
```

### Installation
Currently, only Linux is supported. See releases for latest release.

```zsh
tar -xzf ip2vec-v{VERSION}-bundle.tar.gz --one-top-level
```

This will extract the binaries and the required dependencies to a new folder.
It is important to run the binaries from the `bin/` folder as they are patched
to find dependencies at `$ORIGIN/../lib`.

### Usage

#### Training
```
$> ip2vec-trainer --help
ip2vec-trainer - IP embedding neural network trainer

Usage: ip2vec-trainer [OPTIONS] [DATASET]

Arguments:
  [DATASET]  CSV dataset filepath

Options:
      --src-ip <SRC_IP>                  Column name for source IP
      --dst-ip <DST_IP>                  Column name for destination IP
      --dst-port <DST_PORT>              Column name for destination port
      --protocol <PROTOCOL>              Column name for protocol
  -s, --seed <SEED>                      RNG seed
  -e, --epochs <EPOCHS>                  Epochs to train for
  -r, --ratio <SPLIT_RATIO>              Percentage of dataset to use for training vs. validation
  -b, --batch-size <BATCH_SIZE>          Batch size
  -t, --threads <THREADS>                CPU threads
  -l, --learning-rate <LEARNING_RATE>    Learning rate for gradient adjustment
  -c, --context-window <CONTEXT_WINDOW>  Amount of context examples per sample
  -n, --neg-multiplier <NEG_MULTIPLIER>  Amount of negative context examples per postive context
      --store <STORE>                    Folder path to save configuration and experiment logs
  -h, --help                             Print help
  -V, --version                          Print version
```

#### Inference
```
$> ip2vec --help
ip2vec - IP embedding neural network

Usage: ip2vec --store <STORE> <COMMAND>

Commands:
  single  Single sample inference
  batch   File batch inference
  help    Print this message or the help of the given subcommand(s)

Options:
  -s, --store <STORE>  Path to configuration folder containing model.mpk and config.json
  -h, --help           Print help
  -V, --version        Print version
```
