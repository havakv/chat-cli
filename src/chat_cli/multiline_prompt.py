from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent

bindings = KeyBindings()


# @bindings.add("c-j") # makes alt-j work
@bindings.add("escape", "enter")
def accept_multiline_input(event: KeyPressEvent) -> None:
    """
    On Escape-Enter (Ctrl-J), accept the input.
    or Alt-Enter
    """
    event.current_buffer.validate_and_handle()


def multiline_prompt(prompt: str) -> str:
    session: PromptSession[str] = PromptSession(multiline=True, key_bindings=bindings)
    return session.prompt(prompt)


# def main():
#     print("Enter your input (Press Ctrl-Enter to submit):")
#     try:
#         while True:
#             user_input = session.prompt("> ")
#             print(f"You entered:\n{user_input}")
#     except (KeyboardInterrupt, EOFError):
#         print("Exiting...")
