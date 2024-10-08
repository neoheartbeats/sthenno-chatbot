import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotAdapter
from nonebot import logger as lg

lg.add("errors.log", level="ERROR")

nonebot.init()

driver = nonebot.get_driver()
driver.register_adapter(OneBotAdapter)

# nonebot.load_builtin_plugins("echo")
nonebot.load_plugins("sthenno_chatbot/plugins")

if __name__ == "__main__":
    nonebot.run()
