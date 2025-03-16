function config()
  return {
    scheduler = {},
    gemma = {
      tokenizer = "tokenizer.spm",
      model = "gemma3-4b",
      weights = "4b-it-sfp.sbs"
    },
    session = {
      prefill_tbatch = 64,
      temperature = 0.4,
      top_k = 5
    },
    websocket = {
      max_payload_len = 65536,
      timeout = 300000
    }
  }
end
