from transformers import BlenderbotTokenizer, TFBlenderbotModel
import tensorflow as tf

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = TFBlenderbotModel.from_pretrained("facebook/blenderbot-400M-distill")

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf").input_ids
decoders = tokenizer("Studies show that", return_tensors="tf").input_ids
outputs = model(input_ids=inputs, decoder_input_ids=decoders)

last_hidden_states = outputs.last_hidden_state
