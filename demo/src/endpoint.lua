local res = gemma_loop()
collectgarbage()
ngx.exit(res)
