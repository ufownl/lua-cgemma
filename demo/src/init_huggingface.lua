function config()
  return {
    scheduler = {
      num_threads = 2
    },
    gemma = {
      tokenizer = "tokenizer.spm",
      model = "gemma2-2b-it",
      weights = "2.0-2b-it-sfp.sbs"
    },
    session = {
      prefill_tbatch = 64,
      temperature = 0.4,
      top_k = 50
    },
    websocket = {
      max_payload_len = 65536,
      timeout = 300000
    }
  }
end
