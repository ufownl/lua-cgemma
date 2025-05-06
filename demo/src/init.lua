function config()
  return {
    scheduler = {
      num_threads = 2,
      pin = 0
    },
    gemma = {
      tokenizer = "tokenizer.spm",
      weights = "4b-it-sfp.sbs"
    },
    session = {
      temperature = 0.4,
      top_k = 5
    },
    websocket = {
      max_payload_len = 65536,
      timeout = 300000
    }
  }
end
