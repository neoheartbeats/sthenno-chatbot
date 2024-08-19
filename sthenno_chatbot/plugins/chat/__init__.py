from time import sleep
from nonebot.rule import to_me, command
from nonebot.plugin import on_message, on_command
from nonebot.adapters.onebot.v11 import (
    Bot,
    Event,
    Message,
    MessageSegment,
    MessageEvent,
)
from nonebot import logger as lg
from nonebot import CommandGroup

import openai
import os

from sthenno_chatbot.utils import llm_functions as llm_fn


# Check if message is send by superusers
def superusers_p(bot: Bot, event: Event) -> bool | None:
    superusers = os.getenv("SUPERUSERS")
    if superusers:
        return event.get_user_id() in [int(x) for x in superusers.split(",")]


metioned = on_message(rule=to_me(), priority=10)


message_buffer = []


@metioned.handle()
async def _(event: MessageEvent, bot: Bot):
    global message_buffer

    input_text = event.get_message().extract_plain_text()

    if input_text[0] == ".":
        return

    lg.debug(f"input_text: {input_text}")

    output_text = llm_fn.push_chat_message(
        messages=message_buffer[-10:],
        input_text=input_text,
    )

    if output_text:
        output_text = output_text.strip()
        lg.debug(f"output_text: {output_text}")

        message_buffer.append(llm_fn.message_user(to=input_text))
        message_buffer.append(llm_fn.message_assistant(to=output_text))

        llm_fn.to_chat_file(
            filename="/Users/sthenno/Developer/sthenno-chatbot/conversations.json",
            dt=message_buffer[-2:],
        )

        lg.info(f"conversation: {message_buffer[-2:]} sotred in conversations.json")

        # If there are empty lines, separate the message into multiple segments, then send them separately
        output_text = output_text.strip()
        output_text = output_text.split("\n\n")

        for seg in output_text:
            # Check if seg is not a blank line
            seg = seg.strip()
            if seg:
                await metioned.send(Message(seg))
                sleep(1.5)


# command_group = CommandGroup("sn", prefix_aliases=True)

zeros = on_command("zeros", aliases={"0"}, priority=5)


@zeros.handle()
async def _(event: Event):
    global message_buffer
    message_buffer = []
    await zeros.send(Message("Message buffer is cleaned up."))
