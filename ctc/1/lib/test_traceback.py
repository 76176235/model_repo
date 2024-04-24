import traceback
import log


try:
    child = 0
    a = 1/child
except Exception as e:
    msg = traceback.format_stack()
    bmsg = traceback.format_exc()
    print('print : ', msg)
    print('print : ', "this is test"  + str(bmsg))
    log.error(e)

