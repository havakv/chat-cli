import argparse
import enum
import os
from dataclasses import dataclass
from typing import Generator, Literal

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from typing_extensions import deprecated

from chat_cli import primers
from chat_cli.multiline_prompt import multiline_prompt

RED = "\033[31m"
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
    Gpt4oMini = "gpt-4o-mini"
    Gpt4o = "gpt-4o"

    @staticmethod
    def model_names() -> list[str]:
        return [x.value for x in Model]


class Tracker:
    _client: OpenAI
    messages: list[Msg]
    model: Model
    _primer: Msg | None

    def __init__(self, model: Model) -> None:
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.messages = []
        self.model = model
        self._primer = None

    def prime(self, content: str) -> None:
        self._primer = Msg("system", content)

    def clear(self) -> None:
        self.messages = []

    @deprecated("Use stream_chat")
    def _chat(self, content: str | None) -> Msg:
        if content is not None:
            self.messages.append(Msg("user", content))
        # TODO: add max tokens, etc
        messages = (
            [self._primer.as_chat_completion_message_param()] if self._primer else []
        )
        messages.extend(m.as_chat_completion_message_param() for m in self.messages)
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
        self.messages.append(resp_msg)
        return resp_msg

    def stream_chat(self, content: str | None) -> Generator[str, None, None]:
        if content is not None:
            self.messages.append(Msg("user", content))

        # TODO: add max tokens, etc
        messages = (
            [self._primer.as_chat_completion_message_param()] if self._primer else []
        )
        messages.extend(m.as_chat_completion_message_param() for m in self.messages)
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

        assert (role := next(iter(roles))) == "assistant", f"{roles}"
        self.messages.append(Msg(role, "".join(content_list)))


def input_colored(prompt: str) -> str:
    end_code = "\033[0m"  # Resets the color
    # blue_color = "\033[34m"
    green_color = "\033[32m"
    colored_prompt = f"{green_color}{prompt}{end_code}"
    return input(colored_prompt)


def print_grey(text: str) -> None:
    end_code = "\033[0m"  # Resets the color
    grey = "\033[38;5;249m"  # 250 is same as my terminal and 248 is a bit darker
    print(f"{grey}{text}{end_code}")


def print_content(content: str) -> None:
    end_code = "\033[0m"  # Resets the color
    grey = "\033[38;5;249m"  # 250 is same as my terminal and 248 is a bit darker
    for c in content:
        print(f"{grey}{c}{end_code}", flush=True, end="")


def _process(content: str, tracker: Tracker) -> bool:
    match content:
        case "q" | "exit":
            return True

        case "r" | "reload":
            print("reload")
            tracker.clear()

        case "ml":
            content = multiline_prompt("")
            _process(content, tracker)

        case "debug":
            print(tracker.model)
            for msg in tracker.messages:
                print()
                print(msg)

        case x if x in {"m", "wm"} or x.split()[0] in {"m", "wm"}:
            avail_models = f"Avaiable models: {Model.model_names()}"
            if len(split := x.split()) == 1:
                print(avail_models)
                return False

            if (model := split[1]) not in Model.model_names():
                print(f"No model named {model}")
                print(avail_models)
                return False

            prev_model = tracker.model
            tracker.model = Model(model)

            if tracker.messages:
                if tracker.messages[-1].role == "assistant":
                    tracker.messages.pop()

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
    # TODO: https://cookbook.openai.com/
    tracker = Tracker(Model(args.model))
    while True:
        content = input_colored("\nchat > ")
        if _process(content, tracker):
            break


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
