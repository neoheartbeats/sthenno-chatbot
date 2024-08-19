from pydantic import BaseModel
from nonebot.rule import to_me, command
from nonebot.plugin import on_message, on_command
from nonebot.adapters.onebot.v11 import (
    Bot,
    Event,
    Message,
    MessageSegment,
    MessageEvent,
)
from nonebot.params import CommandArg
from nonebot import logger as lg
from nonebot import CommandGroup

import os
from openai import OpenAI
from multion.client import MultiOn

from sthenno_chatbot.utils import llm_functions as llm_fn
from sthenno_chatbot.utils.file_utils import image_p


_multion_client = MultiOn()


def session(url: str = "https://google.com", local: bool = False):
    s = _multion_client.sessions.create(url=url, local=local)
    return s.session_id


def session_step(session_id: str, cmd: str):
    return _multion_client.sessions.step(session_id=session_id, cmd=cmd)


_llm_client = OpenAI()


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

Example input: Get an image of komeiji_koishi from yande.re/post.
Example output: {"msg": "Get an image of komeiji_koishi.", "url": "https://yande.re/post/"}
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


multion_event = on_command("multion", aliases={"mtn", "🍓"}, priority=10)


@multion_event.handle()
async def _(event: Event, message: Message = CommandArg()):
    if int(event.get_user_id()) != 1829321520:
        return
    prompt = message.extract_plain_text().strip()
    _multion_message = multion_message(prompt)
    if _multion_message:
        if _multion_message.url:
            multion_session_id = session(url=_multion_message.url, local=False)
        else:
            multion_session_id = session(local=False)

        lg.debug(f"multion_session_id: {multion_session_id}")

        if _multion_message.msg:
            _status = "CONTINUE"
            while _status == "CONTINUE":
                _status = session_step(
                    session_id=multion_session_id,
                    cmd=_multion_message.msg,
                ).status
                # TODO: Handle cases if session is not "CONTINUE"
                lg.info(f"[MultiOn] status: {_status}")

                _message = session_step(
                    session_id=multion_session_id,
                    cmd=_multion_message.msg,
                ).message
                if _message is not None:
                    lg.info(f"[MultiOn] message: {_message}")
                    await multion_event.send(Message(f"[🍓] {_message.strip()}"))

            if _message is not None:
                _output_multion_message = multion_message(_message)
                if _output_multion_message:
                    if _output_multion_message.url:
                        # If the output message is an image, send the image
                        if image_p(_output_multion_message.url):
                            await multion_event.send(
                                Message(
                                    f"[CQ:image,file={_output_multion_message.url}]"
                                )
                            )
                    if _output_multion_message.msg:
                        await multion_event.send(
                            Message(
                                f"[🍓] Finished: {_output_multion_message.msg.strip()}"
                            )
                        )


from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

_session = WolframLanguageSession()
if _session.started:
    _session.terminate()


class WolframExpr(BaseModel):
    wolfram_expr: str
    include_graphic: bool | None


def wolfram_expr(prompt: str):
    c = _llm_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an expert at structured data extraction.
You will be given unstructured text and should convert it into the given structure.
If the user's input is not in English, you should translate it to English before converting it.
field wolfram_expr is the expression send to Mathematica.
You can call the WolframAlpha function if real-time information is needed.
You can call LLM such as GPT-4 using the function LLMSynthesize['prompt'].

Example input: What is the integral of x^2?
Example output: {"wolfram_expr": Integrate[x^2, x]}

Example input: Plot x^2.
Example output: {"wolfram_expr": Plot[x^2, {x, -10, 10}], "include_graphic": True}
                """,
            },
            {"role": "user", "content": prompt},
        ],
        response_format=WolframExpr,
    )
    c = c.choices[0].message
    if c.refusal:
        lg.error(f"Refusal: {c.refusal}")
        return
    else:
        return c.parsed


from PIL import Image

mma_event = on_command("mathematica", aliases={"mma"}, priority=10)


@mma_event.handle()
async def _(event: Event, message: Message = CommandArg()):
    if int(event.get_user_id()) != 1829321520:
        return
    msg = message.extract_plain_text().strip()
    if msg:
        expr = wolfram_expr(msg)
        if expr:

            lg.info(f"[MMA] expr: {expr}")
            _session.start()
            if expr.include_graphic:
                # e = _session.evaluate(wlexpr(expr.wolfram_expr))
                # print(e)
                # ex = wl.Export("tmp.png", e, "PNG")
                # _session.evaluate(ex)
                # _session.terminate()
                # await mma_event.send("[CQ:image,file=tmp.png]")
                return
            else:
                e = _session.evaluate(wlexpr(expr.wolfram_expr))
                _session.terminate()
                await mma_event.send(Message(f"[MMA] {e}"))
                return
