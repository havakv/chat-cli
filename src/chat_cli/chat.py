from __future__ import annotations

import argparse
import enum
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Generator, Literal

import pyperclip  # type: ignore
import tiktoken
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from typing_extensions import deprecated

from chat_cli import primers
from chat_cli.multiline_prompt import multiline_prompt

BLUE = "\033[34m"
RED = "\033[31m"
GREEN = "\033[32m"
GREY = "\033[38;5;249m"  # 250 is same as my terminal and 248 is a bit darker
END_CODE = "\033[0m"  # Resets the color


@dataclass
class Msg:
    role: Literal["system", "user", "assistant", "tool"]
    content: str

    def as_chat_completion_message_param(self) -> ChatCompletionMessageParam:
        match self.role:
            case "system":
                return {"role": "system", "content": self.content}
            case "user":
                return {"role": "user", "content": self.content}
            case "assistant":
                return {"role": "assistant", "content": self.content}
            case "tool":
                raise NotImplementedError("Not sure how to handle tool")


# https://openai.com/api/pricing/
class Model(enum.Enum):
    Mini = "gpt-4o-mini"
    Gpt4o = "gpt-4o"
    Large = "gpt-4o-2024-08-06"

    @classmethod
    def try_from_str(cls, name: str) -> Model | None:
        if name in [x.value for x in Model]:
            return Model(name)

        for x in Model:
            if x.name.lower() == name.lower():
                return x

        return None

    @staticmethod
    def model_names() -> list[str]:
        return [f"{x.name.lower()}/{x.value}" for x in Model]

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.value)
        return len(encoding.encode(text))


class Tracker:
    _client: OpenAI
    _messages: list[Msg]
    model: Model
    _primer: Msg | None
    _history: int

    def __init__(self, model: Model) -> None:
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._messages = []
        self.model = model
        self._primer = None
        self._history = 10

    def prime(self, content: str) -> None:
        if content:
            self._primer = Msg("system", content)
        else:
            self._primer = None

    def pop_last(self) -> Msg | None:
        if self._messages:
            self._messages.pop()

    def append(self, msg: Msg) -> None:
        self._messages.append(msg)

    def last(self) -> Msg | None:
        return self._messages[-1] if self._messages else None

    def messages_truncated(self) -> list[Msg]:
        return self._messages[-(self._history * 2) :]

    def clear(self) -> None:
        self._messages = []

    @deprecated("Use stream_chat")
    def _chat(self, content: str | None) -> Msg:
        if content is not None:
            self._messages.append(Msg("user", content))
        # FIXME: add max tokens, etc
        messages = (
            [self._primer.as_chat_completion_message_param()] if self._primer else []
        )
        messages.extend(
            m.as_chat_completion_message_param() for m in self.messages_truncated()
        )
        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model.value,
        )
        assert len(response.choices) == 1, f"{response.choices=}"
        resp = response.choices[0]
        assert resp.finish_reason == "stop", f"{resp=}"
        assert resp.message.role == "assistant", f"{resp.message=}"
        assert resp.message.content is not None, f"{resp.message=}"
        resp_msg = Msg(resp.message.role, resp.message.content)
        self._messages.append(resp_msg)
        return resp_msg

    def stream_chat(self, content: str | None) -> Generator[str, None, None]:
        if content is not None:
            self._messages.append(Msg("user", content))

        # FIXME: add max tokens, etc
        messages = (
            [self._primer.as_chat_completion_message_param()] if self._primer else []
        )
        messages.extend(
            m.as_chat_completion_message_param() for m in self.messages_truncated()
        )
        response = self._client.chat.completions.create(
            messages=messages,
            model=self.model.value,
            stream=True,
        )

        roles: set[str] = set()
        content_list: list[str] = []

        for chunk in response:
            assert chunk.choices, "choices should never be zero I think"
            choice = chunk.choices[0]

            if choice.delta.role:
                roles.add(choice.delta.role)
                assert len(roles) < 2, f"{roles}"

            content = choice.delta.content
            if content is None:
                assert choice.finish_reason == "stop", f"{chunk}"
                break

            content_list.append(content)
            yield content

        role = next(iter(roles))
        assert role == "assistant", f"{roles}"
        self._messages.append(Msg(role, "".join(content_list)))


def input_colored(prompt: str) -> str:
    colored_prompt = f"{GREEN}{prompt}{END_CODE}"
    return input(colored_prompt)


def print_grey(text: str) -> None:
    print(f"{GREY}{text}{END_CODE}")


def print_content(content: str) -> None:
    for c in content:
        # print(f"{GREY}{c}{END_CODE}", flush=True, end="")  # noqa: ERA001
        print(c, flush=True, end="")


def _extract_code_blocks(content: str) -> list[tuple[str, str]]:
    code_block_pattern = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)

    blocks: list[tuple[str, str]] = []
    for match in code_block_pattern.finditer(content):
        shell = match.group(1)
        code = match.group(2)
        assert not ((shell is None) ^ (code is None))
        if shell and code:
            blocks.append((shell, code))

    return blocks


# FIXME:
#  - limit history used in queries
#  - cli directly: c: tell a joke
#  - prompt for short concise answers
#  - chat shell
#  - copy last answer
#  - vim integration:
#       - copy answer to vim
#       - write question in vim
#  - set temperature
#  - set seed
#  - set max token length
def _process(content: str, tracker: Tracker) -> bool:
    content = content.strip()
    match content:
        case "":
            return False

        case "q" | "exit":
            return True

        case "r" | "reload":
            print("reload")
            tracker.clear()

        case "yy" | "yank":
            if last := tracker.last():
                content = last.content.strip()
                pyperclip.copy(content)
            else:
                print("not messages to copy")

        case "yc":
            if last := tracker.last():
                code_blocks = _extract_code_blocks(last.content)
                if len(code_blocks) != 1:
                    print(f"found {len(code_blocks)=}, can't execute")

                _shell, content = code_blocks[0]
                pyperclip.copy(content)
            else:
                print("not messages to copy")

        case "x":
            if last := tracker.last():
                # Execute the code using the specified shell
                code_blocks = _extract_code_blocks(last.content)
                if len(code_blocks) != 1:
                    print(f"found {len(code_blocks)=}, can't execute")
                    return False

                shell, code = code_blocks[0]
                assert "sudo" not in code
                result = subprocess.run(
                    shell,
                    input=code,
                    text=True,
                    capture_output=True,
                    shell=True,
                    check=False,
                )

                print(f"\n{BLUE}system > {END_CODE}", flush=True)
                print_content(result.stdout)
                print_content(result.stderr)
                tracker.append(Msg("system", f"{result.stdout}{result.stderr}"))

        case x if x.startswith("x "):
            code = x.removeprefix("x ")
            assert "sudo" not in code
            result = subprocess.run(
                code, text=True, capture_output=True, shell=True, check=False
            )

            print(f"\n{BLUE}system > {END_CODE}", flush=True)
            print_content(result.stdout)
            print_content(result.stderr)
            tracker.append(Msg("system", f"{result.stdout}{result.stderr}"))

        case "ml":
            content = multiline_prompt("")
            _process(content, tracker)

        case "debug":
            print(tracker.model)
            print(tracker._primer)  # type: ignore
            for msg in tracker.messages_truncated():
                print()
                print(f"tokens: {tracker.model.num_tokens(msg.content)}")
                print(msg)

        case x if x in {"m", "wm"} or x.split()[0] in {"m", "wm"}:
            avail_models = f"Available models: {Model.model_names()}"
            if len(split := x.split()) == 1:
                print(avail_models)
                return False

            if not (model := Model.try_from_str(split[1])):
                print(f"No model named '{split[1]}'")
                print(avail_models)
                return False

            prev_model = tracker.model
            tracker.model = Model(model)

            if tracker.messages_truncated():
                if tracker.messages_truncated()[-1].role == "assistant":
                    tracker.pop_last()

                # FIXME: make this the same as the case _:
                print(flush=True)
                print(f"{RED}{tracker.model.value} > {END_CODE}", end="", flush=True)
                for response in tracker.stream_chat(None):
                    print_content(response)
                print(flush=True)

            if split[0] == "wm":
                tracker.model = prev_model

        case _:
            print(flush=True)
            print(f"{RED}{tracker.model.value} > {END_CODE}", end="", flush=True)
            for response in tracker.stream_chat(content):
                print_content(response)
            print(flush=True)

    return False


def chat(args: argparse.Namespace) -> None:
    # FIXME: https://cookbook.openai.com/
    tracker = Tracker(Model(args.model))
    tracker.prime(primers.CHAT)
    while True:
        try:
            content = input_colored("\nchat > ")
            if _process(content, tracker):
                break
        except KeyboardInterrupt:
            # useful for clearing lines
            pass


# FIXME: could use structured outputs https://cookbook.openai.com/examples/structured_outputs_intro
#       probably wouldn't work with streaming, but could still be useful.
def synonyms(args: argparse.Namespace) -> None:
    tracker = Tracker(Model(args.model))
    tracker.prime(primers.SYNONYMS)
    while not _process(input_colored("\nsynonyms > "), tracker):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatGPT tools")
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", choices=Model.model_names()
    )

    subparsers = parser.add_subparsers(title="subcommands", dest="command")

    synonyms_parser = subparsers.add_parser("synonyms", help="list synonyms")
    synonyms_parser.set_defaults(func=synonyms)

    args = parser.parse_args()
    if args.command is None:
        return chat(args)

    return args.func(args)
