require("vips")

function config()
  return {
    scheduler = {
      num_threads = 6,
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
    vlm_mode = {
      max_file_size = 1024 * 1024 * 4,
      resize_to = 896
    },
    websocket = {
      max_payload_len = 1024 * 1024 * 8,
      timeout = 300000
    }
  }
end
