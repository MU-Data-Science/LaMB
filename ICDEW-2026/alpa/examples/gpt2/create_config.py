from transformers import GPT2Config

config = GPT2Config.from_pretrained("gpt2", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, vocab_size=50256)

# Setting the parameters of the model; larger models can FAIL with Alpa due to memory
config.n_head = 2
config.n_layer = 2
config.n_embd = 512
config.n_positions = 512
config.n_ctx = 512

config.save_pretrained("./wikipedia-gpt2-small")
