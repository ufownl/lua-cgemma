local _M = {}

function _M.parse(cli_args)
  local args = {}
  for i, v in ipairs(cli_args) do
    if string.sub(v, 1, 2) == "--" then
      if cli_args[i + 1] and string.sub(cli_args[i + 1], 1, 2) ~= "--" then
        args[string.sub(v, 3)] = cli_args[i + 1]
      else
        args[string.sub(v, 3)] = true
      end
    end
  end
  return args
end

function _M.help(description, usage)
  require("cgemma").info()
  print()
  print(description)
  print()
  print(string.format("Usage: %s", usage))
  print()
  print("Available options:")
  print("  --tokenizer: Path of tokenizer model file. (default: tokenizer.spm)")
  print("  --model: Model type (default: 2b-it)")
  print("    2b-it = 2B parameters, instruction-tuned")
  print("    7b-it = 7B parameters instruction-tuned")
  print("    9b-it = 9B parameters instruction-tuned")
  print("    27b-it = 27B parameters instruction-tuned")
  print("    gr2b-it = griffin 2B parameters, instruction-tuned")
  print("  --weights: Path of model weights file. (default: 2b-it-sfp.sbs)")
  print("  --weight_type: Weight type (default: sfp)")
end

return _M
