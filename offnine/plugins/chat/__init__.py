from time import sleep

import ujson as json
from nonebot import logger as lg
from nonebot.adapters.onebot.v11 import Bot, Event, Message, MessageEvent
from nonebot.plugin import on_command, on_message
from nonebot.rule import to_me
from openai import OpenAI
from urlextract import URLExtract

from offnine.utils import llm_functions as llm_fn
from offnine.utils.file_utils import image_p

message_pair_index = 0
message_buffer = []

thoughts = []

metioned = on_message(rule=to_me(), priority=10)

call_id = ""

extractor = URLExtract()

import os
import ssl
import tempfile
from io import BytesIO

import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

chat = llm_fn.Chat()


@metioned.handle()
async def _(event: MessageEvent, bot: Bot):

    text_in = event.get_message().extract_plain_text()

    if text_in == ".0":
        return

    current_agent: str = chat.get_agent()

    completion = chat.chat(text_in)

    if completion:
        for seg in completion:
            if "assistant" in seg:
                seg = f"[{current_agent} → {json.loads(seg).get('assistant')}]"
            if "Query" in seg or "Result: " in seg or "Playing" in seg:
                continue
            segs: list[str] = str(seg).strip().split("\n\n")
            for s in segs:
                await metioned.send(message=Message(message=s))
                sleep(1.5)


import subprocess

zeros = on_command("zeros", aliases={"0"}, priority=5)


@zeros.handle()
async def _(event: Event):
    chat.reset()
    subprocess.run(["pkill", "mpv"])
    await zeros.finish(message=Message(message="Chat session reset."))
