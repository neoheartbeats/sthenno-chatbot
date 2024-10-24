# Standard library imports
import copy
import json
from collections import defaultdict
from typing import Callable, List, Union

# Package/library imports
from openai import OpenAI

from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)

# Local imports
from .util import debug_print, function_to_json, merge_chunk

__CTX_VARS_NAME__ = "context_variables"


class Swarm:
    def __init__(self, client=None):
        if not client:
            client = OpenAI()
        self.client = client

    def get_chat_completion(
        self,
        agent: Agent,
        history: list,
        context_variables: dict,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history
        debug_print(debug, "Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]

        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        create_params = {
            "model": agent.model,
            "messages": messages,
            "stream": stream,
        }

        if tools:
            create_params["tools"] = tools
            create_params["tool_choice"] = agent.tool_choice or "auto"
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        if agent.model == "sthenno":  # TODO
            create_params["temperature"] = 0.75
            create_params["top_p"] = 0.85
            create_params["max_tokens"] = 1024
            create_params["extra_body"] = dict(repetition_penalty=1.05)

        return self.client.chat.completions.create(**create_params)  # type: ignore

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"""Failed to cast response to string: {result}.
Make sure agent functions return a string or Result object. Error: {str(e)}."""
                    debug_print(debug, error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        functions: list[AgentFunction],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            id = tool_call.id[-10:]  # last 10 characters of id
            tool_call.id = id
            # handle missing tool case, skip to next tool
            if name not in function_map:
                debug_print(debug, f"Tool {name} not found in function map.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": id,  # TODO
                        "name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            debug_print(debug, f"Processing tool call: {name} with arguments {args}")

            func = function_map[name]
            # pass context_variables to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = function_map[name](**args)

            result: Result = self.handle_function_result(raw_result, debug)
            id = tool_call.id[-10:]  # last 10 characters of id
            tool_call.id = id
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": id,  # TODO
                    "name": name,
                    "content": result.value,
                }
            )
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: list,
        context_variables: dict = {},
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        raise NotImplementedError("Stream mode not implemented yet.")
        # active_agent = agent
        # context_variables = copy.deepcopy(context_variables)
        # history = copy.deepcopy(messages)
        # init_len = len(messages)

        # while len(history) - init_len < max_turns:

        #     message = {
        #         "content": "",
        #         # "sender": agent.name,
        #         "role": "assistant",
        #         "function_call": None,
        #         "tool_calls": defaultdict(
        #             lambda: {
        #                 "function": {"arguments": "", "name": ""},
        #                 "id": "",
        #                 "type": "",
        #             }
        #         ),
        #     }

        #     # get completion with current history, agent
        #     completion = self.get_chat_completion(
        #         agent=active_agent,
        #         history=history,
        #         context_variables=context_variables,
        #         stream=True,
        #         debug=debug,
        #     )

        #     yield {"delim": "start"}
        #     for chunk in completion:
        #         delta = json.loads(chunk.choices[0].delta.json())
        #         # if delta["role"] == "assistant":
        #         # delta["sender"] = active_agent.name
        #         yield delta
        #         delta.pop("role", None)
        #         # delta.pop("sender", None)
        #         merge_chunk(message, delta)
        #     yield {"delim": "end"}

        #     message["tool_calls"] = list(message.get("tool_calls", {}).values())
        #     if not message["tool_calls"]:
        #         message["tool_calls"] = None
        #     debug_print(debug, "Received completion:", message)
        #     history.append(message)

        #     if not message["tool_calls"] or not execute_tools:
        #         debug_print(debug, "Ending turn.")
        #         break

        #     # convert tool_calls to objects
        #     tool_calls = []
        #     for tool_call in message["tool_calls"]:
        #         function = Function(
        #             arguments=tool_call["function"]["arguments"],
        #             name=tool_call["function"]["name"],
        #         )
        #         tool_call_object = ChatCompletionMessageToolCall(
        #             id=tool_call["id"][-10:],
        #             function=function,
        #             type=tool_call["type"],  # TODO
        #         )
        #         # TODO
        #         tool_calls.append(tool_call_object)

        #     # handle function calls, updating context_variables, and switching agents
        #     partial_response = (
        #         self.handle_tool_calls(
        #             tool_calls,
        #             active_agent.functions,
        #             context_variables,
        #             debug,
        #         )
        #         if tool_calls
        #         else Response()
        #     )
        #     history.extend(partial_response.messages)
        #     context_variables.update(partial_response.context_variables)
        #     if partial_response.agent:
        #         active_agent = partial_response.agent

        # yield {
        #     "response": Response(
        #         messages=history[init_len:],
        #         agent=active_agent,
        #         context_variables=context_variables,
        #     )
        # }

    def run(
        self,
        agent: Agent,
        messages: list,
        context_variables: dict = {},
        debug: bool = False,
        max_turns: int | float = float("inf"),
        execute_tools: bool = True,
    ) -> Response:

        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                stream=False,
                debug=debug,
            )
            message = completion.choices[0].message  # type: ignore
            debug_print(debug, "Received completion:", message)
            message_dict = json.loads(message.model_dump_json())
            history.append(dict(role=message.role))
            if message.content:
                history[-1]["content"] = message.content
            if message.tool_calls:
                history[-1]["tool_calls"] = message_dict["tool_calls"]
                history[-1]["tool_calls"][0]["id"] = message_dict["tool_calls"][0][
                    "id"
                ][-10:]
            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response: Response = self.handle_tool_calls(
                tool_calls=message.tool_calls,
                functions=active_agent.functions,
                context_variables=context_variables,
                debug=debug,
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
