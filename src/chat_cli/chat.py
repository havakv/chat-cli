import argparse
import enum
import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from chat_cli import primers


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


class Model(enum.Enum):
    Gpt4oMini = "gpt-4o-mini"
    Gpt4o = "gpt-4o"

    @staticmethod
    def model_names() -> list[str]:
        return [x.value for x in Model]


class Tracker:
    _client: OpenAI
    _messages: list[Msg]
    model: Model

    def __init__(self, model: Model) -> None:
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._messages = []
        self.model = model

    def _send(self) -> ChatCompletion:
        # TODO: add max tokens, etc
        response = self._client.chat.completions.create(
            messages=[m.as_chat_completion_message_param() for m in self._messages],
            model=self.model.value,
        )
        return response

    def prime(self, content: str) -> None:
        self._messages.append(Msg("system", content))

    def clear(self) -> None:
        self._messages = []

    def chat(self, content: str | None) -> Msg:
        if content is not None:
            self._messages.append(Msg("user", content))
        response = self._send()
        assert len(response.choices) == 1, f"{response.choices=}"
        resp = response.choices[0]
        assert resp.finish_reason == "stop", f"{resp=}"
        assert resp.message.role == "assistant", f"{resp.message=}"
        assert resp.message.content is not None, f"{resp.message=}"
        resp_msg = Msg(resp.message.role, resp.message.content)
        self._messages.append(resp_msg)
        return resp_msg


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


def chat(args: argparse.Namespace) -> None:
    tracker = Tracker(Model(args.model))
    while True:
        match content := input_colored("\nchat > "):
            case "q" | "exit":
                return

            case "r" | "reload":
                print("reload")
                tracker.clear()

            case x if x.startswith("wm ") or x.startswith("m "):
                avail_models = f"Avaiable models: {Model.model_names()}"
                if len(split := x.split()) == 1:
                    print(avail_models)
                    continue

                if (model := split[1]) not in Model.model_names():
                    print(f"No model named {model}")
                    print(avail_models)
                    continue

                prev_model = tracker.model
                tracker.model = Model(model)
                resp = tracker.chat(None)
                print_grey(f"\n{resp.content}")

                if split[0] == "wm":
                    tracker.model = prev_model

            case _:
                resp = tracker.chat(content)
                print_grey(f"\n{resp.content}")


def synonyms(args: argparse.Namespace) -> None:
    tracker = Tracker(Model(args.model))
    tracker.prime(primers.SYNONYMS)
    while (content := input("\nsynonyms > ")) != "q":
        resp = tracker.chat(content)
        print(f"\n{resp.content}")


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
