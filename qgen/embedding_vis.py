import io
import logging
import os
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm


# The code below is mostly from `https://www.tensorflow.org/programmers_guide/embedding`.

def prepare_embeddings_for_tensorboard(embeddings_path: str, log_dir: str,
                                       token_filter: Optional[set] = None):
    """
    Prepare embeddings for TensorBoard by writing them to `log_dir` in the required format.
    :param embeddings_path: The path to the GloVe embeddings file.
    :param log_dir: The directory for TensorBoard.
    :param token_filter: The set of tokens to use. If not given, then all tokens will be used.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.info("Loading embeddings from `%s`.", embeddings_path)

    metadata = []
    vectors = []

    with open(embeddings_path, encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading embeddings",
                         mininterval=2, unit=" tokens", unit_scale=True):
            split = line.rstrip().split(' ')
            token = split[0]
            if token_filter is None or token in token_filter:
                metadata.append(token)
                vector = [float(x) for x in split[1:]]
                vectors.append(vector)

    assert len(vectors) > 0, "No vectors found."

    embeddings = np.array(vectors, dtype=np.float32)

    # Write metadata.
    metadata_path = os.path.join(log_dir, 'vocab.tsv')
    with io.open(metadata_path, 'w', encoding='utf-8') as f:
        for message in metadata:
            f.write(message)
            f.write("\n")

    embedding_var = tf.Variable(embeddings, name='message_embedding')

    with tf.Session() as sess:
        saver = tf.train.Saver([embedding_var])
        sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(log_dir, 'embeddings.ckpt'))

    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = os.path.join(log_dir, metadata_path)

    summary_writer = tf.summary.FileWriter(log_dir)

    projector.visualize_embeddings(summary_writer, config)

    logging.info("Run: `tensorboard --logdir=\"%s\"` to see the embeddings.", log_dir.replace("\\", "\\\\"))


if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(filename)s::%(funcName)s\n%(message)s',
                        level=logging.INFO)

    this_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Prepare embeddings for visualization.")
    parser.add_argument('--embeddings_path', type=str,
                        default=os.path.join(this_dir, 'data', 'glove.6B', 'glove.6B.100d.txt'),
                        help="Path to the embeddings.")
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(this_dir, 'log_dir'),
                        help="Directory to save the embeddings to for TensorBoard.")

    args = parser.parse_args()

    prepare_embeddings_for_tensorboard(args.embeddings_path, args.log_dir)
