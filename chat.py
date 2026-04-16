from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4
import asyncio
from PIL import Image as PImage
from game_interface_and_emoji_trivia_prototype import *
from nicegui import ui


# -----------------------------
# NiceGUI chat helpers
# -----------------------------


@dataclass
class ChatItem:
    user_id: str
    avatar: str
    text: str
    stamp: str
    images: list[str]


chat_log: list[ChatItem] = []


def now_stamp() -> str:
    return datetime.now().strftime("%X")


def pil_to_data_url(img: PImage.Image) -> str:
    return f"data:image/jpeg;base64,{encode_image_base64(img)}"


def format_prompt_from_game(game: MyGame) -> tuple[str, list[str]]:
    prompt = game.get_prompt_data()
    images: list[str] = []

    q_lines = []
    if prompt["query_alphabet"].type == 0:
        # q_lines.append(
        #     f"**Query ({prompt['query_alphabet'].alphabet_name}):** {prompt['query']}"
        # )
        q_lines.append(f"{prompt['query']}")
    else:
        # q_lines.append(f"**Query ({prompt['query_alphabet'].alphabet_name}):**")
        for idx in list(str(prompt["query"])):
            img = prompt["query_alphabet"].indexed_letters[idx]
            images.append(pil_to_data_url(img))

    a_lines = []
    if prompt["task_alphabet"].type == 0:
        # a_lines.append(
        #     f"**Alphabet ({prompt['task_alphabet'].alphabet_name}):** {prompt['task_alphabet'].indexed_letters}"
        # )
        pass
    else:
        # a_lines.append(
        #     f"**Alphabet ({prompt['task_alphabet'].alphabet_name}):** image index pairs"
        # )
        # for idx, img in prompt["task_alphabet"].indexed_letters.items():
        #     a_lines.append(f"- {idx}")
        for idx, img in prompt["task_alphabet"].indexed_letters.items():
            images.append(pil_to_data_url(img))

    text = "\n".join(
        [
            prompt["description"],
            "",
            # f"**{prompt['signature_name']}**",
            # "",
            *q_lines,
            "",
            *a_lines,
            "",
            "_Type your answer and click Send._",
        ]
    )
    return text, images


def post(user_id: str, avatar: str, text: str, images: list[str] | None = None) -> None:
    chat_log.append(
        ChatItem(
            user_id=user_id,
            avatar=avatar,
            text=text,
            stamp=now_stamp(),
            images=images or [],
        )
    )
    chat_messages.refresh()


@ui.refreshable
def chat_messages(own_id: str) -> None:
    if not chat_log:
        ui.label("No messages yet").classes("mx-auto my-36")
        return

    for m in chat_log:
        ui.chat_message(
            text=m.text,
            stamp=m.stamp,
            avatar=m.avatar,
            sent=(m.user_id == own_id),
        )

        if m.images:
            sent = m.user_id == own_id
            with ui.row().classes("w-full"):
                if sent:
                    ui.element("div").classes("flex-grow")
                with ui.row().classes("flex-wrap gap-2 max-w-full"):
                    for data_url in m.images:
                        ui.image(data_url).classes(
                            "w-24 h-24 object-contain rounded shadow"
                        )
                if not sent:
                    ui.element("div").classes("flex-grow")

    ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")
