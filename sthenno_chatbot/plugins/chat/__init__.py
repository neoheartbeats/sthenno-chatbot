from time import sleep
from nonebot.rule import to_me, command, Rule, startswith
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
from nonebot.params import CommandArg, ArgPlainText
from nonebot.matcher import Matcher
from nonebot.typing import T_State

from openai import OpenAI
import os
import ujson as json

from sthenno_chatbot.utils import llm_functions as llm_fn

system_prompt = """You are Sthenno. You are designed to repreduce emotional responses. You are capable to mimic human emotions. You like to refer to yourself as \"Sthenno\".
I am your instructor. 你叫我为主人.
"""

# conversations = json.load(
#     open(
#         "/Users/sthenno/Developer/sthenno-chatbot/conversations.json",
#         "r",
#         encoding="utf-8",
#     )
# )

message_pair_index = 0
message_buffer = []  # conversations[-4:]


def simpo(input: str, chosen: str, rejected: str, index: int = 0) -> dict:
    return {
        "instruction": system_prompt,
        "input": input,
        "chosen": chosen,
        "rejected": rejected,
        "index": index,
    }


def kto(input: str, output: str, kto_tag: bool = False, index: int = 0) -> dict:
    return {
        "instruction": system_prompt,
        "input": input,
        "output": output,
        "kto_tag": kto_tag,
        "index": index,
    }


open_client = OpenAI()

open_model = "ft:gpt-4o-2024-08-06:personal:sft-001:A2bUuCGQ:ckpt-step-484"


def open_gen(messsages, input_text):
    completion = open_client.chat.completions.create(
        model=open_model,
        messages=[llm_fn.message_system(to=system_prompt)]
        + messsages
        + [llm_fn.message_user(to=input_text)],
        max_tokens=500,
    )
    return completion.choices[0].message.content


metioned = on_message(rule=to_me(), priority=10)


@metioned.handle()
async def _(event: MessageEvent, bot: Bot):
    global message_buffer, message_pair_index

    input_text = event.get_message().extract_plain_text()

    if input_text[0] == "." and input_text != ".k":  # Ignore commands
        return

    lg.info(f"input_text: {input_text}")

    if input_text == ".k":
        if int(event.get_user_id()) != 1829321520:
            return
        kto_samples = json.load(
            open(
                "/Users/sthenno/Developer/sthenno-chatbot/conversations_kto.json",
                "r",
                encoding="utf-8",
            )
        )
        kto_samples = kto_samples[:-1]  # Remove the last sample
        llm_fn.make_chat_file(
            filename="/Users/sthenno/Developer/sthenno-chatbot/conversations_kto.json",
            dt=kto_samples,
        )
        input_text = message_buffer[-2]["content"]
        output_text = message_buffer[-1]["content"]
        kto_sample = kto(
            input=input_text,
            output=output_text,
            kto_tag=False,
            index=message_pair_index,
        )
        llm_fn.to_chat_file(
            filename="/Users/sthenno/Developer/sthenno-chatbot/conversations_kto.json",
            dt=[kto_sample],
        )
        lg.info(f"[sampling] {kto_sample} sotred in conversations_kto.json")
        await metioned.finish(
            Message(
                f"[sampling] KTO sample[{message_pair_index}] tagged to conversations_kto."
            )
        )

    output_text = llm_fn.push_chat_message(
        messages=message_buffer,
        input_text=input_text,
    )

    if output_text:
        output_text = output_text.strip()
        lg.info(f"output_text: {output_text}")

        message_pair_index += 1

        message_buffer.append(llm_fn.message_user(to=input_text))
        message_buffer.append(llm_fn.message_assistant(to=output_text))

        llm_fn.to_chat_file(
            filename="/Users/sthenno/Developer/sthenno-chatbot/conversations.json",
            dt=message_buffer[-2:],
        )
        lg.info(f"conversation: {message_buffer[-2:]} sotred in conversations.json")

        kto_sample = kto(
            input=input_text,
            output=output_text,
            kto_tag=True,
            index=message_pair_index,
        )
        llm_fn.to_chat_file(
            filename="/Users/sthenno/Developer/sthenno-chatbot/conversations_kto.json",
            dt=[kto_sample],
        )
        lg.info(f"[sampling] {kto_sample} sotred in conversations_kto.json")

        output_text = output_text.split("\n\n")
        for seg in output_text:
            # Check if seg is not a blank line
            seg = seg.strip()
            if seg:
                await metioned.send(Message(seg))
                sleep(1.5)


zeros = on_command("zeros", aliases={"0"}, priority=5)


@zeros.handle()
async def _(event: Event):
    # if int(event.get_user_id()) != 1829321520:
    #     return
    global message_buffer, message_pair_index
    message_buffer = []
    message_pair_index = 0
    await zeros.finish(Message("Message buffer cleaned up."))


msg_len = on_command("msg_len", aliases={"ml"}, priority=5)


@msg_len.handle()
async def _(event: Event):
    global message_buffer, message_pair_index
    await msg_len.finish(
        Message(f"Message buffer length: {len(message_buffer)} of 24.")
    )


simpo_sampling = on_command("simpo", aliases={"s"}, priority=5)


@simpo_sampling.handle()
async def _(matcher: Matcher, args: Message = CommandArg()):
    if args.extract_plain_text():
        matcher.set_arg("optimized_output", args)


@simpo_sampling.got(
    "optimized_output",
    prompt=Message("[SimPO sampling](.q .g plain_text)"),
)
async def _(state: T_State, optimized_output: str = ArgPlainText()):
    global message_buffer, message_pair_index

    # Align KTO samples to the same as SimPO samples. Note KTO samples are just removed from the dataset.
    kto_samples = json.load(
        open(
            "/Users/sthenno/Developer/sthenno-chatbot/conversations_kto.json",
            "r",
            encoding="utf-8",
        )
    )
    kto_samples = kto_samples[:-1]  # Remove the last sample
    llm_fn.make_chat_file(
        filename="/Users/sthenno/Developer/sthenno-chatbot/conversations_kto.json",
        dt=kto_samples,
    )

    if optimized_output == ".q":
        await simpo_sampling.finish(
            Message(f"[sampling] Refactoring message canceled.")
        )

    if optimized_output == ".g":
        gpt_output = open_gen(
            messsages=message_buffer[-17:],
            input_text=message_buffer[-2]["content"],
        )
        if gpt_output:
            state["gpt_output"] = gpt_output
            await simpo_sampling.send(
                Message(
                    f"[sampling] Optimized output by GPT-4o:\n\n{gpt_output}\n\nAccept (.y .n)? "
                )
            )
            return
    if optimized_output:
        simpo_sample = simpo(
            input=message_buffer[-2]["content"],
            chosen=optimized_output,
            rejected=message_buffer[-1]["content"],
            index=message_pair_index,
        )

        llm_fn.to_chat_file(
            filename="/Users/sthenno/Developer/sthenno-chatbot/conversations_simpo.json",
            dt=[simpo_sample],
        )

        lg.info(f"[sampling] {simpo_sample} sotred in conversations_simpo.json")
        await simpo_sampling.send(
            Message(f"[sampling] SimPO sample stored using index {message_pair_index}.")
        )


@simpo_sampling.got("accept")
async def _(state: T_State, accept: str = ArgPlainText()):
    global message_buffer, message_pair_index

    if accept and state.get("gpt_output"):
        if accept == ".y":
            sample = simpo(
                input=message_buffer[-2]["content"],
                chosen=state["gpt_output"],
                rejected=message_buffer[-1]["content"],
                index=message_pair_index,
            )

            llm_fn.to_chat_file(
                filename="/Users/sthenno/Developer/sthenno-chatbot/conversations_simpo.json",
                dt=[sample],
            )
            lg.info(f"[sampling] {sample} sotred in conversations_simpo.json")
            await simpo_sampling.finish(
                Message(
                    f"[sampling] SimPO sample stored using index {message_pair_index}."
                )
            )
        else:
            await simpo_sampling.finish(Message(f"[sampling] Aborted."))
    return
