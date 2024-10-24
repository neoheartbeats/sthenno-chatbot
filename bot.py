import nonebot
from nonebot import logger
from nonebot.adapters.onebot.v11 import Adapter as OneBotAdapter

logger.add("errors.log", level="ERROR")

nonebot.init()

driver = nonebot.get_driver()
driver.register_adapter(OneBotAdapter)

nonebot.load_plugins("offnine/plugins")

if __name__ == "__main__":
    nonebot.run()
