local ok, err = require("cgemma").compress_weights("2b-it", "2b-it.sbs", "2b-it-sfp.sbs")
if not ok then
  print("Opoos! ", err)
  return
end
print("Done.")