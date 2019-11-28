from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import os
import time
import pickle

# Batch size
BATCH_SIZE = 1

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024
EPOCHS=20
char2idx_file_name = "./saved_data/char2idx.txt"
idx2char_file_name = "./saved_data/idx2char.txt"
vocabsize_file_name = "./saved_data/vocab_size.txt"
directory_name = "saved_data"
checkpoint_dir = "./training_checkpoints"
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
	model = tf.keras.Sequential([
		tf.keras.layers.Embedding(vocab_size, embedding_dim,
								  batch_input_shape=[batch_size, None]),
		tf.keras.layers.GRU(rnn_units,
							return_sequences=True,
							stateful=True,
							recurrent_initializer='glorot_uniform'),
		tf.keras.layers.Dense(vocab_size)
	  ])
	return model
def loss(labels, logits):
	return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
  
def generate_text(char2idx, idx2char, model, start_string):
	# Evaluation step (generating text using the learned model)

	# Number of characters to generate
	num_generate = 1000

	# Converting our start string to numbers (vectorizing)
	input_eval = [char2idx[s] for s in start_string]
	input_eval = tf.expand_dims(input_eval, 0)

	# Empty string to store our results
	text_generated = []

	# Low temperatures results in more predictable text.
	# Higher temperatures results in more surprising text.
	# Experiment to find the best setting.
	temperature = 1.0

	# Here batch size == 1
	model.reset_states()
	for i in range(num_generate):
		predictions = model(input_eval)
		# remove the batch dimension
		predictions = tf.squeeze(predictions, 0)

		# using a categorical distribution to predict the word returned by the model
		predictions = predictions / temperature
		predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

		# We pass the predicted word as the next input to the model
		# along with the previous hidden state
		input_eval = tf.expand_dims([predicted_id], 0)

		text_generated.append(idx2char[predicted_id])

	return (start_string + ''.join(text_generated))
def model_train(checkpoint_prefix, filename):
	
	text = open(filename).read()
	
	vocab = sorted(set(text))

	char2idx = {u:i for i, u in enumerate(vocab)}
	idx2char = np.array(vocab)
	text_as_int = np.array([char2idx[c] for c in text])
	# The maximum length sentence we want for a single input in characters
	seq_length = 100
	examples_per_epoch = len(text)//(seq_length+1)

	# Create training examples / targets
	char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
	sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
	dataset = sequences.map(split_input_target)
	dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
	
	# Length of the vocabulary in chars
	vocab_size = len(vocab)

	model = build_model(
	vocab_size = len(vocab),
	embedding_dim=embedding_dim,
	rnn_units=rnn_units,
	batch_size=BATCH_SIZE)
	for input_example_batch, target_example_batch in dataset.take(1):
		example_batch_predictions = model(input_example_batch)
		#print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
	model.summary()	
	sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
	sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
	model.compile(optimizer='adam', loss=loss)
	# Directory where the checkpoints will be saved
	checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_prefix,
	save_weights_only=True)

	history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
	if not os.path.exists(directory_name):
		os.makedirs(directory_name)
	with open(char2idx_file_name, 'wb') as handle:
		pickle.dump(char2idx, handle)
		
	with open(idx2char_file_name, 'wb') as handle:
		pickle.dump(idx2char, handle)

	
	with open(vocabsize_file_name, "w") as text_file:
		text_file.write("{}".format(vocab_size))
	return model
	
	
def reload_model (checkpoint_dir, vocab_size, embedding_dim, rnn_units):

	model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
	model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
	model.build(tf.TensorShape([1, None]))
	return model

def create_article(start_string):
	with open(char2idx_file_name, 'rb') as handle:
		char2idx = pickle.loads(handle.read())
	with open(idx2char_file_name, 'rb') as handle:
		idx2char = pickle.loads(handle.read())
	with open(vocabsize_file_name, "r") as text_file:
		vocab_size = [int(x) for x in text_file]
		vocab_size = vocab_size[0]
	model = reload_model(checkpoint_dir, vocab_size, embedding_dim, rnn_units)
	return generate_text(char2idx, idx2char, model, start_string=start_string)
	
def main():
	filename = "examples_thai.txt"
	model = reload_model(checkpoint_dir, vocab_size, embedding_dim, rnn_units)
	print(generate_text(char2idx, idx2char, model, start_string=u"Dieselnoi "))
	return 0

#main()