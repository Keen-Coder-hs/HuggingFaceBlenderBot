from transformers import BlenderbotTokenizer, FlaxBlenderbotForConditionalGeneration, BlenderbotConfig

mname = "facebook/blenderbot-400M-distill"
model = FlaxBlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)
UTTERANCE = "My friends are cool but they eat too many carbs."
print("Human: ", UTTERANCE)

inputs = tokenizer([UTTERANCE], max_length=1024, return_tensors='np')
reply_ids = model.generate(**inputs).sequences
print("Bot: ", [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in reply_ids])

import jax.numpy as jnp
from transformers import BlenderbotTokenizer, FlaxBlenderbotForConditionalGeneration

model = FlaxBlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

text = "My friends are cool but they eat too many carbs."
inputs = tokenizer(text, max_length=1024, return_tensors="jax")
encoder_outputs = model.encode(**inputs)

decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

outputs = model.decode(decoder_input_ids, encoder_outputs)
logits = outputs.logits
print(logits)
