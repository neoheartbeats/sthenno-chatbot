from pydantic import BaseModel
from nonebot.rule import to_me, command
from nonebot.plugin import on_message, on_command
from nonebot.adapters.onebot.v11 import (
    Bot,
    Event,
    Message,
    MessageEvent,
)
from nonebot.params import CommandArg
from nonebot import logger as lg
from nonebot import CommandGroup

import openai
import multion
from multion.client import MultiOn

from sthenno_chatbot.utils import llm_functions as llm_fn
from sthenno_chatbot.utils.file_utils import image_p


_multion_client = MultiOn()


def session(url: str = "https://google.com", local: bool = False):
    s = _multion_client.sessions.create(url=url, local=local)
    return s.session_id


def session_step(session_id: str, cmd: str):
    return _multion_client.sessions.step(session_id=session_id, cmd=cmd)


_llm_client = openai.OpenAI()


class MultiOnMessage(BaseModel):
    msg: str
    url: str | None


def multion_message(prompt: str):
    c = _llm_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an expert at structured data extraction.
You will be given unstructured text from a research paper and should convert it into the given structure.

Example input: 从 pixiv.com 找一张古明地恋的图片.
Example output: {"msg": "找一张古明地恋的图片.", "url": "https://pixiv.com"}
                """,
            },
            {"role": "user", "content": prompt},
        ],
        response_format=MultiOnMessage,
    )
    multion_message = c.choices[0].message
    if multion_message.refusal:
        lg.error(f"Refusal: {multion_message.refusal}")
        return
    else:
        return multion_message.parsed


multion_event = on_command("mtn", aliases={"multion"}, priority=10)


@multion_event.handle()
async def _(event: Event, message: Message = CommandArg()):
    if int(event.get_user_id()) != 1829321520:
        return
    prompt = message.extract_plain_text().strip()
    _multion_message = multion_message(prompt)
    if _multion_message:
        if _multion_message.url:
            multion_session_id = session(url=_multion_message.url, local=True)
        else:
            multion_session_id = session(local=True)

        lg.debug(f"multion_session_id: {multion_session_id}")

        if _multion_message.msg:
            _status = "CONTINUE"
            while _status == "CONTINUE":
                _status = session_step(
                    session_id=multion_session_id,
                    cmd=_multion_message.msg,
                ).status
                lg.info(f"[MultiOn] status: {_status}")

                _message = session_step(
                    session_id=multion_session_id,
                    cmd=_multion_message.msg,
                ).message
                if _message is not None:
                    lg.info(f"[MultiOn] message: {_message}")
                    await multion_event.send(Message(f"[MultiOn]: {_message}"))

            if _message is not None:
                _output_multion_message = multion_message(_message)
                if _output_multion_message:
                    if _output_multion_message.url:
                        if image_p(_output_multion_message.url):
                            await multion_event.send(
                                Message(
                                    f"[CQ:image,file={_output_multion_message.url}]"
                                )
                            )
                    if _output_multion_message.msg:
                        await multion_event.send(
                            Message(f"[MultiOn][DONE]: {_output_multion_message.msg}")
                        )
