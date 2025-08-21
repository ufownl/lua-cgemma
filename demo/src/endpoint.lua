local ws, err = require("resty.websocket.server"):new(config().websocket)
if not ws then
  ngx.log(ngx.ERR, "websocket error: ", err)
  return ngx.HTTP_CLOSE
end
local ok, err = pcall(gemma_loop, ws)
ws:send_close()
collectgarbage()
if not ok then
  ngx.log(ngx.ERR, err)
  ngx.exit(ngx.ERROR)
end
ngx.exit(ngx.OK)
