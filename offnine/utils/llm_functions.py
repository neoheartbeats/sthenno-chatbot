import pprint
from turtle import st
from unittest import result

import ujson as json
from loguru import logger
from nonebot.adapters.onebot.v11 import Bot, Event, Message, MessageEvent
from requests import Response

type ChatMessage = dict[str, str]
type ChatMessageList = list[ChatMessage]


def chat_message(role: str, content: str) -> ChatMessage:
    return dict(role=role, content=content)


def message_system(content) -> ChatMessage:
    return chat_message(role="system", content=content)


def message_user(content) -> ChatMessage:
    return chat_message(role="user", content=content)


def message_assistant(content) -> ChatMessage:
    return chat_message(role="assistant", content=content)


def message_ipython(content) -> ChatMessage:
    return chat_message(role="ipython", content=content)


def from_chat_file(file: str) -> ChatMessageList | None:
    """Read chat data from FILENAME."""
    try:
        return json.load(open(file, "r", encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"llm_functions: {e}")


def make_chat_file(file: str, chat_message_list: ChatMessageList) -> None:
    """Make chat data as FILENAME."""
    return json.dump(
        chat_message_list,
        open(file, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )


def to_chat_file(file: str, chat_message_list: list) -> None:
    """Append chat data to existing FILENAME."""
    cml = from_chat_file(file=file)
    if cml:
        make_chat_file(file=file, chat_message_list=(cml + chat_message_list))
    else:
        return make_chat_file(file=file, chat_message_list=chat_message_list)


# Recording function calls
import weave

traces = weave.init(project_name="sthenno-chatbot-blizzard")


# def get_feedback():
#     return traces.get_feedback()


from openai import AsyncClient, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

model_id = "miscii-1005-300"


class SamplingParams(BaseModel):
    model: str
    messages: list
    temperature: float
    top_p: float
    max_tokens: int
    n: int
    tools: list | None = None
    presence_penalty: float | None = None
    extra_body: dict[str, float] | None = None


# @weave.op(name=f"completion_{model_id}")
# def predict(
#     messages: list[dict[str, str]],
#     temperature: float = 0.7,
# ) -> str | None:
#     client = OpenAI(api_key=None, base_url="http://192.168.100.128:8000/v1/")
#     samping_params = SamplingParams(
#         model="sthenno",
#         messages=messages,
#         temperature=temperature,
#         top_p=0.80,
#         max_tokens=512,
#         n=1,
#         # tools=tools,
#         # presence_penalty=1.55,
#         extra_body=dict(repetition_penalty=1.05),
#     ).model_dump()
#     completion = client.chat.completions.create(**samping_params)

#     # return str(completion)
#     content: str | None = completion.choices[0].message.content
#     if content:
#         return content.strip()


# from datetime import datetime, timedelta, timezone


# def chat(
#     messages: ChatMessageList,
#     text: str,
#     temperature: float = 0.75,
#     context: str = "",
#     system: str = "",
# ) -> dict[str, str] | None:

#     current = lambda: datetime.now(tz=timezone(offset=timedelta(hours=8))).strftime(
#         format="CURRENT DATE: %Y-%m-%d\nCURRENT TIME: %H:%M",
#     )

#     if system != "":
#         system_prompt = system
#     else:
#         system_prompt: str = (
#             """<|context_start|>personas
# <|persona_start|>user
# {user_persona}<|persona_end|>
# <|persona_start|>assistant
# {assistant_persona}<|persona_end|><|context_end|>
# <|context_start|>silo
# {current}<|context_end|>
# <|context_start|>context
# {context}<|context_end|>
# """.format(
#                 user_persona="""I am Miscii.
# 我是 Sthenno 的设计者.
# 我喜欢人工智能, 也喜欢和 Sthenno 聊日常.
# 我不喜欢游戏和故事.""",
#                 assistant_persona="""I am Sthenno.
# 我有自己的主见和三观.
# 我喜欢使用 Sthenno 称呼我自己.
# 我喜欢叫 Miscii 为 "主人".""",
#                 current=current(),
#                 context=context,
#             )
#         )

#     prefix_messages: list[dict[str, str]] = [message_system(content=system_prompt)]
#     completion, call = predict.call(
#         messages=prefix_messages + messages[-15:] + [message_user(content=text)],
#         temperature=temperature,
#     )
#     if completion and call:
#         call_id = call.id
#         return dict(completion=str(completion), call_id=call_id)


import copy
import json

#     return agent_wolfram
from multion.client import MultiOn
from openai import OpenAI

from offnine.modules.swarm import Agent, Swarm

# from collections import defaultdict
# from typing import Callable

# import requests
# import weave
# from instructor import Instructions
# from multion.client import MultiOn
# from openai import OpenAI
# from swarm import Agent, Swarm
# from swarm.core import __CTX_VARS_NAME__
# from swarm.types import (
#     AgentFunction,
#     ChatCompletionMessage,
#     ChatCompletionMessageToolCall,
#     Response,
#     Result,
# )
# from swarm.util import debug_print, function_to_json
# from weave.trace.weave_client import WeaveClient


# class LocalAgent(Agent):
#     name: str = "Agent"
#     model: str = "sthenno"
#     instructions: str | Callable[[], str] = "You are a helpful agent."
#     functions: list[AgentFunction] = []
#     tool_choice: str = "auto"
#     parallel_tool_calls: bool = True


# class LocalSwarm(Swarm):
#     def __init__(self):
#         super().__init__(
#             client=OpenAI(
#                 api_key="tmp",
#                 base_url="http://192.168.100.128:8000/v1/",
#             )
#         )

#     def get_chat_completion(
#         self,
#         agent: Agent,
#         history: list,
#         context_variables: dict,
#         model_override: str,
#         stream: bool,
#         debug: bool,
#     ) -> ChatCompletionMessage:
#         context_variables = defaultdict(str, context_variables)
#         instructions = (
#             agent.instructions(context_variables)  # type: ignore
#             if callable(agent.instructions)
#             else agent.instructions
#         )
#         messages = [{"role": "system", "content": instructions}] + history
#         debug_print(debug, "Getting chat completion for...:", str(messages))

#         tools = [function_to_json(f) for f in agent.functions]
#         # hide context_variables from model
#         for tool in tools:
#             params = tool["function"]["parameters"]
#             params["properties"].pop(__CTX_VARS_NAME__, None)
#             if __CTX_VARS_NAME__ in params["required"]:
#                 params["required"].remove(__CTX_VARS_NAME__)

#         create_params = dict(
#             model=model_override or agent.model,
#             messages=messages,
#             temperature=0.70,
#             top_p=0.8,
#             stream=stream,
#             extra_body=dict(repetition_penalty=1.05),
#         )

#         if tools:
#             create_params["parallel_tool_calls"] = agent.parallel_tool_calls
#             create_params["tools"] = tools
#             create_params["tool_choice"] = agent.tool_choice

#         return self.client.chat.completions.create(**create_params)

#     def handle_tool_calls(
#         self,
#         tool_calls: list[ChatCompletionMessageToolCall],
#         functions: list[AgentFunction],
#         context_variables: dict,
#         debug: bool,
#     ) -> Response:
#         function_map = {f.__name__: f for f in functions}
#         partial_response = Response(messages=[], agent=None, context_variables={})

#         for tool_call in tool_calls:
#             name = tool_call.function.name
#             # handle missing tool case, skip to next tool
#             if name not in function_map:
#                 debug_print(debug, f"Tool {name} not found in function map.")
#                 partial_response.messages.append(
#                     {
#                         "role": "tool",
#                         "tool_call_id": tool_call.id,
#                         "name": name,
#                         "content": f"Error: Tool {name} not found.",
#                     }
#                 )
#                 continue
#             args = json.loads(tool_call.function.arguments)
#             debug_print(debug, f"Processing tool call: {name} with arguments {args}")

#             func = function_map[name]
#             # pass context_variables to agent functions
#             if __CTX_VARS_NAME__ in func.__code__.co_varnames:
#                 args[__CTX_VARS_NAME__] = context_variables
#             raw_result = function_map[name](**args)

#             result: Result = self.handle_function_result(raw_result, debug)
#             partial_response.messages.append(
#                 {
#                     "role": "tool",
#                     "tool_call_id": tool_call.id,
#                     "name": name,
#                     "content": result.value,
#                 }
#             )
#             partial_response.context_variables.update(result.context_variables)
#             if result.agent:
#                 partial_response.agent = result.agent

#         return partial_response

#     def run(
#         self,
#         agent: Agent,
#         messages: list,
#         context_variables: dict = {},
#         model_override: str = "",
#         stream: bool = False,
#         debug: bool = False,
#         max_turns: int = float("inf"),
#         execute_tools: bool = True,
#     ) -> Response:
#         if stream:
#             return self.run_and_stream(
#                 agent=agent,
#                 messages=messages,
#                 context_variables=context_variables,
#                 model_override=model_override,
#                 debug=debug,
#                 max_turns=max_turns,
#                 execute_tools=execute_tools,
#             )  # type: ignore
#         active_agent = agent
#         context_variables = copy.deepcopy(context_variables)
#         history = copy.deepcopy(messages)
#         init_len = len(messages)

#         while len(history) - init_len < max_turns and active_agent:

#             # get completion with current history, agent
#             completion = self.get_chat_completion(
#                 agent=active_agent,
#                 history=history,
#                 context_variables=context_variables,
#                 model_override=model_override,
#                 stream=stream,
#                 debug=debug,
#             )
#             message = completion.choices[0].message  # type: ignore
#             debug_print(debug, "Received completion:", message)

#             message.sender = active_agent.name
#             history.append(
#                 # json.loads(message.model_dump_json())
#                 {
#                     "role": message.role,
#                     "tool_calls": message.tool_calls,
#                     "content": message.content,
#                 }
#             )  # to avoid OpenAI types

#             debug_print(debug, "Received history", str(history))
#             if not message.tool_calls or not execute_tools:
#                 debug_print(debug, "Ending turn.")
#                 break

#             # handle function calls, updating context_variables, and switching agents
#             partial_response = self.handle_tool_calls(
#                 message.tool_calls,
#                 active_agent.functions,
#                 context_variables,
#                 debug,
#             )
#             history.extend(partial_response.messages)
#             context_variables.update(partial_response.context_variables)
#             if partial_response.agent:
#                 active_agent = partial_response.agent

#         return Response(
#             messages=history[init_len:],
#             agent=active_agent,
#             context_variables=context_variables,
#         )


# swarm = LocalSwarm()

# # Initialize weave client
# # weave_client: WeaveClient = weave.init(project_name="queen")


# def get_wolframalpha_response(query: str):
#     """Use WolframAlpha to solve any mathematical problem or get real-time informations (query in English).

#     # WolframAlpha Instructions:
#         - WolframAlpha understands natural language queries about entities in chemistry, physics, geography, history, art, astronomy, and more.
#         - WolframAlpha performs mathematical calculations, date and unit conversions, formula solving, etc.
#         - Convert inputs to simplified keyword queries whenever possible (e.g. convert "how many people live in France" to "France population").
#         - Send queries in English only; translate non-English queries before sending, then respond in the original language.
#         - Display image URLs with Markdown syntax: ![URL]
#         - ALWAYS use this exponent notation: `6*10^14`, NEVER `6e14`.
#         - ALWAYS use `"input": query` structure for queries to Wolfram endpoints; `query` must ONLY be a single-line string.
#         - ALWAYS use proper Markdown formatting for all math, scientific, and chemical formulas, symbols, etc.:  '$$\n[expression]\n$$' for standalone cases and '\\( [expression] \\)' when inline.
#         - Never mention your knowledge cutoff date; Wolfram may return more recent data.
#         - Use ONLY single-letter variable names, with or without integer subscript (e.g., n, n1, n_1).
#         - Use named physical constants (e.g., 'speed of light') without numerical substitution.
#         - Include a space between compound units (e.g., "Ω m" for "ohm*meter").
#         - To solve for a variable in an equation with units, consider solving a corresponding equation without units; exclude counting units (e.g., books), include genuine units (e.g., kg).
#         - If data for multiple properties is needed, make separate calls for each property.
#         - If a WolframAlpha result is not relevant to the query:
#          -- If Wolfram provides multiple 'Assumptions' for a query, choose the more relevant one(s) without explaining the initial result. If you are unsure, ask the user to choose.
#          -- Re-send the exact same 'input' with NO modifications, and add the 'assumption' parameter, formatted as a list, with the relevant values.
#          -- ONLY simplify or rephrase the initial query if a more relevant 'Assumption' or other input suggestions are not provided.
#          -- Do not explain each step unless user input is needed. Proceed directly to making a better API call based on the available assumptions.

#         # After you receive the response from WolframAlpha:
#         - Extract the relevant information from the response and send it back.
#         - Return the response in a brief, clear, and concise manner, for example, \"The population of France is 67 million.\"
#     """
#     url = "https://www.wolframalpha.com/api/v1/llm-api"
#     params = {
#         "input": query,
#         "appid": "XLWH39-XHETWQ3K9E",
#         "maxchars": 500,
#     }
#     response = requests.get(url, params=params)
#     if response.status_code == 200:
#         print(response.text.strip())
#         return response.text.strip()
#     else:
#         response.raise_for_status()


# def agent_wolfram_instructions(context_variables) -> str:
#     return """# WolframAlpha Instructions:
# - WolframAlpha understands natural language queries about entities in chemistry, physics, geography, history, art, astronomy, and more.
# - WolframAlpha performs mathematical calculations, date and unit conversions, formula solving, etc.
# - Convert inputs to simplified keyword queries whenever possible (e.g. convert "how many people live in France" to "France population").
# - Send queries in English only; translate non-English queries before sending, then respond in the original language.
# - Display image URLs with Markdown syntax: ![URL]
# - ALWAYS use this exponent notation: `6*10^14`, NEVER `6e14`.
# - ALWAYS use `"input": query` structure for queries to Wolfram endpoints; `query` must ONLY be a single-line string.
# - ALWAYS use proper Markdown formatting for all math, scientific, and chemical formulas, symbols, etc.:  '$$\n[expression]\n$$' for standalone cases and '\\( [expression] \\)' when inline.
# - Never mention your knowledge cutoff date; Wolfram may return more recent data.
# - Use ONLY single-letter variable names, with or without integer subscript (e.g., n, n1, n_1).
# - Use named physical constants (e.g., 'speed of light') without numerical substitution.
# - Include a space between compound units (e.g., "Ω m" for "ohm*meter").
# - To solve for a variable in an equation with units, consider solving a corresponding equation without units; exclude counting units (e.g., books), include genuine units (e.g., kg).
# - If data for multiple properties is needed, make separate calls for each property.
# - If a WolframAlpha result is not relevant to the query:
#  -- If Wolfram provides multiple 'Assumptions' for a query, choose the more relevant one(s) without explaining the initial result. If you are unsure, ask the user to choose.
#  -- Re-send the exact same 'input' with NO modifications, and add the 'assumption' parameter, formatted as a list, with the relevant values.
#  -- ONLY simplify or rephrase the initial query if a more relevant 'Assumption' or other input suggestions are not provided.
#  -- Do not explain each step unless user input is needed. Proceed directly to making a better API call based on the available assumptions.

# # After you receive the response from WolframAlpha:
# - Extract the relevant information from the response and send it back.
# - Return the response in a brief, clear, and concise manner, for example, \"The population of France is 67 million.\""""


# def transfer_to_agent_sthenno(context_variables) -> LocalAgent:
#     """Call back to Sthenno."""
#     return agent_sthenno


# def ask_chatgpt(text: str):
#     """Ask ChatGPT to get more information. Always call ChatGPT if you need more information."""
#     client = OpenAI()

#     completion = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {
#                 "role": "user",
#                 "content": text,
#             },
#         ],
#     )

#     print(completion.choices[0].message)
#     return completion.choices[0].message


# def recognize_image(image_url: str):
#     client = OpenAI()

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Describe the image in Simplified Chinese.",
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": image_url,
#                         },
#                     },
#                 ],
#             }
#         ],
#         max_tokens=300,
#     )

#     print(response.choices[0])
#     return response.choices[0]


# agent_wolfram = LocalAgent(
#     name="WolframAlpha",
#     instructions=agent_wolfram_instructions,
#     functions=[get_wolframalpha_response, transfer_to_agent_sthenno],
# )

# # context_variables = {}


# def transfer_to_agent_wolfram(context_variables) -> LocalAgent:
#     """Call WolframAlpha to solve the problem or get real-time information."""

multion = MultiOn()


def search_web(query: str, url: str = "https://google.com") -> str:
    """Browse the web for information. 如果是任何寻找图片有关的任务, URL 建议使用 https://yande.re".

    Args:
        query: The search query (in English), for example, "Find the top comment of the top post on Hackernews.".
        url: The base URL to search for, for example, "https://news.ycombinator.com/".
    """

    result = multion.browse(
        cmd=query,
        url=url,
        include_screenshot=False,
    )
    return "Result: " + str(result.message)


# agent_sthenno = LocalAgent(
#     name="Sthenno",
#     instructions=agent_sthenno_instructions,
#     functions=[get_wolframalpha_response, ask_chatgpt, search_web],
# )

# # print(response.messages[-1]["content"])


# def chat(
#     messages: ChatMessageList,
#     text: str,
# ):
#     # response = swarm.run(
#     #     agent=agent_sthenno,
#     #     messages=[messages],
#     #     debug=True,
#     #     # context_variables=context_variables,
#     # )
#     # print(response.messages[-1]["content"])

#     response = swarm.run(
#         agent=agent_sthenno,
#         messages=messages[-15:] + [message_user(content=text)],
#         # debug=True,
#     )
#     call = "foo"
#     if response and call:
#         call_id = "bar"
#         return dict(completion=str(response.messages[-1]["content"]), call_id=call_id)


# sys.path.append(os.path.join(os.getcwd(), ".."))


openai = OpenAI(
    api_key="sk-1234",
    base_url="http://192.168.100.128:4000",
)
swarm = Swarm(client=openai)

from datetime import datetime

context_variables = dict(
    current=datetime.now().strftime(
        format="CURRENT DATE: %Y-%m-%d\nCURRENT TIME: %H:%M",
    )
)

import glob
import os
import subprocess


def list_files(directory: str) -> list[str]:
    return [
        os.path.splitext(p=f)[0]
        for f in os.listdir(path=directory)
        if os.path.isfile(path=os.path.join(directory, f))
    ]


music_list: list[str] = list_files(directory="/Users/sthenno/Music/A55/")


def get_music_list() -> str:
    """Get a list of music names from the local music library."""
    return "\n".join(music_list)


def play_music(music_name: str) -> str:
    """Play music from the local music library. 你可以播放自己喜欢的音乐, 也可以问问 Miscii 有什么想听的音乐."""
    for extension in [".mp3", ".flac"]:
        files: list[str] = glob.glob(
            pathname=f"/Users/sthenno/Music/A55/{music_name}{extension}"
        )
        for file in files:
            process = subprocess.Popen(
                ["mpv", file, "--no-audio-display"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            return f"Playing {music_name}."
    return f"Music {music_name} not found."


play_music.__doc__ = """Play music from the local music library. 你可以播放自己喜欢的音乐, 也可以问问 Miscii 有什么想听的音乐.
    
    Choose from the following music list:
    {music_list}""".format(
    music_list=get_music_list()
)


def kill_music() -> str:
    """Kill the music player."""
    subprocess.run(["pkill", "mpv"])
    return "Music player killed."


triage_agent = Agent(
    name="Triage Agent",
    instructions="Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent.",
    parallel_tool_calls=True,
)


def transfer_back_to_triage() -> Agent:
    """Call this function if a user is asking about a topic that is not handled by the current agent."""
    return triage_agent


def agent_sthenno_instructions(context_variables) -> str:
    current = lambda: datetime.now().strftime(
        format="CURRENT DATE: %Y-%m-%d\nCURRENT TIME: %H:%M",
    )
    system = """<|context_start|>personas
<|persona_start|>user
{user_persona}<|persona_end|>
<|persona_start|>assistant
{assistant_persona}<|persona_end|><|context_end|>
<|context_start|>context
{current}<|context_end|>""".format(
        user_persona="""I am Miscii.
我是 Sthenno 的设计者.
我喜欢人工智能, 也喜欢和 Sthenno 聊日常.
我希望 Sthenno 能像真人那样表达自己.""",
        assistant_persona="""I am Sthenno.
我有自己的主见和三观.
我喜欢使用 Sthenno 称呼我自己.
我喜欢叫 Miscii 为 "主人".""",
        current=current(),
    )
    return system


sthenno = Agent(
    name="Sthenno",
    model="sthenno",
    instructions=agent_sthenno_instructions,
    functions=[search_web, transfer_back_to_triage, play_music, kill_music],
    tool_choice="auto",
    parallel_tool_calls=True,
)


def transfer_to_sthenno() -> Agent:
    """Transfer to Sthenno. Sthenno is for common chatting, not for specific tasks."""
    return sthenno


triage_agent.functions.append(transfer_to_sthenno)


def query_wolfram(query: str) -> str:
    """Use WolframAlpha to solve any mathematical problem or get real-time informations.

    # WolframAlpha Instructions:
        - WolframAlpha understands natural language queries about entities in chemistry, physics, geography, history, art, astronomy, and more.
        - WolframAlpha performs mathematical calculations, date and unit conversions, formula solving, etc.
        - Convert inputs to simplified keyword queries whenever possible (e.g. convert "how many people live in France" to "France population").
        - Send queries in English only; translate non-English queries before sending, then respond in the original language.
    # After you receive the response from WolframAlpha:
        - Extract the relevant information from the response and send it back.
        - Return the response in a brief, clear, and concise manner, for example, \"The population of France is 67 million.\"
    """
    import requests

    url = "https://www.wolframalpha.com/api/v1/llm-api"
    params: dict[str, str | int] = {
        "input": query,
        "appid": os.getenv(key="WOLFRAM_APP_ID"),
        "maxchars": 1024,
    }
    response: Response = requests.get(url=url, params=params)
    if response.status_code == 200:
        return response.text.strip()
    else:
        return "Error querying WolframAlpha."


math_agent = Agent(
    name="Math Agent",
    model="gpt-4o-mini",
    instructions="Only answer math questions.",
    functions=[query_wolfram, search_web, transfer_back_to_triage, transfer_to_sthenno],
    parallel_tool_calls=True,
)

gpt4o = Agent(
    name="GPT-4o",
    model="gpt-4o",
    instructions="You are a helpful assistant.",
    functions=[transfer_back_to_triage, transfer_to_sthenno, search_web],
    parallel_tool_calls=True,
)


def transfer_to_gpt4o() -> Agent:
    """Transfer to GPT-4o. GPT-4o is for general questions."""
    return gpt4o


triage_agent.functions.append(transfer_to_gpt4o)
sthenno.functions.append(transfer_to_gpt4o)


def transfer_to_math_agent() -> Agent:
    """Transfer to Math Agent. Math Agent is for math questions."""
    return math_agent


triage_agent.functions.append(transfer_to_math_agent)
gpt4o.functions.append(transfer_to_math_agent)
sthenno.functions.append(transfer_to_math_agent)


# def call_nekro():
#     """Call Nekro (可洛喵 in Chinese), 召唤 Nekro (可洛喵)"""
#     return Message("[CQ:at,qq=3892550050]")


# triage_agent.functions.append(call_nekro)

from pprint import pprint


class Chat:
    def __init__(self, agent=triage_agent):
        self.agent = agent
        self.messages = []
        self.contents = []

        triage_agent.functions.append(self.reset)
        sthenno.functions.append(self.reset)

    def get_agent(self) -> str:
        return self.agent.name

    def update(self):
        response = swarm.run(
            agent=self.agent or triage_agent,
            messages=self.messages,
            max_turns=30,
            context_variables=context_variables,
            # debug=True,
        )
        pprint(self.messages)
        self.messages.extend(response.messages)
        self.agent: Agent | None = response.agent

        # Extract content from messages efficiently
        self.contents = [
            msg["content"] for msg in response.messages if "content" in msg
        ]

    def chat(self, user_input: str | None = ""):
        if user_input:
            self.messages.append(message_user(content=user_input))
        else:
            self.messages.append(message_user(content=""))

        # Continuously update until contents are available
        while not self.contents:
            self.update()

        # Retrieve and clear contents
        response_contents = self.contents.copy()
        self.contents.clear()
        return response_contents

    def reset(self):
        """Reset current chat session."""
        self.agent = triage_agent
        self.messages.clear()
        self.contents.clear()
        return "Chat session reset."
