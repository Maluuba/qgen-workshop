#!/usr/bin/env python

import itertools
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.layers.core import Dense
from tqdm import tqdm, trange

from qgen.embedding_vis import prepare_embeddings_for_tensorboard
from qgen.data import training_data, test_data, collapse_documents, expand_answers
from qgen.embedding import embeddings_path, glove, look_up_token, UNKNOWN_TOKEN, START_TOKEN, END_TOKEN

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log_dir')

prepare_embeddings_for_tensorboard(embeddings_path, log_dir)

embedding = tf.get_variable("embedding", initializer=glove)

EMBEDDING_DIMENS = glove.shape[1]


document_tokens = tf.placeholder(tf.int32, shape=[None, None], name="document_tokens")
document_lengths = tf.placeholder(tf.int32, shape=[None], name="document_lengths")

document_emb = tf.nn.embedding_lookup(embedding, document_tokens)

forward_cell = GRUCell(EMBEDDING_DIMENS)
backward_cell = GRUCell(EMBEDDING_DIMENS)

answer_outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, document_emb, document_lengths, dtype=tf.float32, scope="answer_rnn")
answer_outputs = tf.concat(answer_outputs, 2)

answer_tags = tf.layers.dense(inputs=answer_outputs, units=2)

answer_labels = tf.placeholder(tf.int32, shape=[None, None], name="answer_labels")

answer_mask = tf.sequence_mask(document_lengths, dtype=tf.float32)
answer_loss = seq2seq.sequence_loss(logits=answer_tags, targets=answer_labels, weights=answer_mask, name="answer_loss")


encoder_input_mask = tf.placeholder(tf.float32, shape=[None, None, None], name="encoder_input_mask")
encoder_inputs = tf.matmul(encoder_input_mask, answer_outputs, name="encoder_inputs")
encoder_lengths = tf.placeholder(tf.int32, shape=[None], name="encoder_lengths")

encoder_cell = GRUCell(forward_cell.state_size + backward_cell.state_size)

_, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs, encoder_lengths, dtype=tf.float32, scope="encoder_rnn")


decoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name="decoder_inputs")
decoder_labels = tf.placeholder(tf.int32, shape=[None, None], name="decoder_labels")
decoder_lengths = tf.placeholder(tf.int32, shape=[None], name="decoder_lengths")

decoder_emb = tf.nn.embedding_lookup(embedding, decoder_inputs)
helper = seq2seq.TrainingHelper(decoder_emb, decoder_lengths)

projection = Dense(embedding.shape[0], use_bias=False)

decoder_cell = GRUCell(encoder_cell.state_size)

decoder = seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection)

decoder_outputs, _, _ = seq2seq.dynamic_decode(decoder, scope="decoder")
decoder_outputs = decoder_outputs.rnn_output

question_mask = tf.sequence_mask(decoder_lengths, dtype=tf.float32)
question_loss = seq2seq.sequence_loss(logits=decoder_outputs, targets=decoder_labels, weights=question_mask, name="question_loss")


loss = tf.add(answer_loss, question_loss, name="loss")
tf.summary.scalar("loss", loss)
optimizer = tf.train.AdamOptimizer().minimize(loss)

merged = tf.summary.merge_all()


saver = tf.train.Saver()
session = tf.InteractiveSession()
writer = tf.summary.FileWriter(log_dir, session.graph)

EPOCHS = 5

epoch = 0
for i in range(1, EPOCHS + 1):
    if os.path.exists("model-{0}.index".format(i)):
        epoch = i

if epoch:
    saver.restore(session, "model-{0}".format(epoch))
else:
    session.run(tf.global_variables_initializer())

batch_index = 0
batch_count = None
for epoch in trange(epoch + 1, EPOCHS + 1, desc="Epochs", unit="epoch"):
    batches = tqdm(training_data(), total=batch_count, desc="Batches", unit="batch")
    for batch in batches:
        _, loss_value, summary = session.run([optimizer, loss, merged], {
            document_tokens: batch["document_tokens"],
            document_lengths: batch["document_lengths"],
            answer_labels: batch["answer_labels"],
            encoder_input_mask: batch["answer_masks"],
            encoder_lengths: batch["answer_lengths"],
            decoder_inputs: batch["question_input_tokens"],
            decoder_labels: batch["question_output_tokens"],
            decoder_lengths: batch["question_lengths"],
        })
        batches.set_postfix(loss=loss_value)
        writer.add_summary(summary, batch_index)
        writer.flush()
        batch_index += 1

    if batch_count is None:
        batch_count = batch_index

    saver.save(session, "model", epoch)


batch = next(test_data())
batch = collapse_documents(batch)

answers = session.run(answer_tags, {
    document_tokens: batch["document_tokens"],
    document_lengths: batch["document_lengths"],
})
answers = np.argmax(answers, 2)


batch = expand_answers(batch, answers)

helper = seq2seq.GreedyEmbeddingHelper(embedding, tf.fill([batch["size"]], START_TOKEN), END_TOKEN)
decoder = seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection)
decoder_outputs, _, _ = seq2seq.dynamic_decode(decoder, maximum_iterations=16)
decoder_outputs = decoder_outputs.rnn_output

questions = session.run(decoder_outputs, {
    document_tokens: batch["document_tokens"],
    document_lengths: batch["document_lengths"],
    answer_labels: batch["answer_labels"],
    encoder_input_mask: batch["answer_masks"],
    encoder_lengths: batch["answer_lengths"],
})
questions[:,:,UNKNOWN_TOKEN] = 0
questions = np.argmax(questions, 2)

for i in range(batch["size"]):
    question = itertools.takewhile(lambda t: t != END_TOKEN, questions[i])
    print("Question: " + " ".join(look_up_token(token) for token in question))
    print("Answer: " + batch["answer_text"][i])
    print()
