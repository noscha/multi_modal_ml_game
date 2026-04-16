from uuid import uuid4
from pathlib import Path
import asyncio

from nicegui import ui

from chat import *
from game_interface_and_emoji_trivia_prototype import *


async def ai_submit_until_valid(
    game: MyGame,
    own_id: str,
    bot_id: str,
    bot_avatar: str,
) -> bool:
    while True:
        agent, desc = game.current_agent_and_description()

        post(bot_id, bot_avatar, "🤖 AI is thinking...")

        response = await asyncio.to_thread(
            agent.adapter.present_data,
            game.round_state,
            type(agent.signature).NAME,
            desc,
        )
        query = await asyncio.to_thread(agent.adapter.get_data, response)

        query = str(query).strip()
        post(bot_id, bot_avatar, f"(AI) {query}")

        result = game.submit(query)
        post(bot_id, bot_avatar, result["message"])

        if result["accepted"]:
            return result["done"]

        post(bot_id, bot_avatar, "⚠️ AI answer invalid, retrying...")


@ui.page("/")
async def main():
    own_id = str(uuid4())
    user_avatar = f"https://robohash.org/{own_id}?bgset=bg2"
    bot_id = "GAME"
    bot_avatar = "https://robohash.org/notebook-game?bgset=bg1"

    # clear old chat on refresh
    chat_log.clear()

    # same setup as your notebook
    validator_alphabet = Alphabet("Points", ["0", "1"], 0)
    letter_alphabet = Alphabet("lower_letters", list("abcdefghijklmnopqrstuvwxyz"), 0)
    country_alphabet = Alphabet("countries", ["USA", "Germany", "France"], 0)

    image_alphabet = Alphabet(
        "images",
        {
            "0": load_local_image(Path("images/logoA.jpg")),
            "1": load_local_image(Path("images/logoB.jpg")),
            "2": load_local_image(Path("images/titanic.jpg")),
            "3": load_local_image(Path("images/usa_1.jpg")),
            "4": load_local_image(Path("images/usa_2.jpg")),
            "5": load_local_image(Path("images/usa_3.jpg")),
            "6": load_local_image(Path("images/usa_4.jpg")),
        },
        1,
    )

    description_4i1w = (
        "Choose 4 images which represent the country the best",
        "Choose the country best described by the four images",
        "Is this the country you had in mind?",
    )

    examples_4i1w = [
        Example("USA", image_alphabet, country_alphabet, validator_alphabet),
        Example("France", image_alphabet, country_alphabet, validator_alphabet),
    ]
    task_4i1w = Task(examples_4i1w, description_4i1w)


    image_alphabet = Alphabet(
        "images",
        {
            "0": load_local_image(Path("images/A.jpg")),
            "1": load_local_image(Path("images/B.jpg")),
        },
        1,  # dict → images
    )

    text_alphabet = Alphabet(
        "captions",
        list("abcdefghijklmnopqrstuvwxyz "),
        0,  # list[str]
    )

    validator_alphabet = Alphabet(
        "Points",
        ["0", "1"],
        0,
    )

    example = Example(
        message="01",
        encoder_alphabet=text_alphabet,
        decoder_alphabet=image_alphabet,
        validator_alphabet=validator_alphabet,
    )

    description = (
        "Look at the images and write a caption describing ONE of them",
        "To which image belongs the following caption",
        "Is the selected image correct? (1 = yes, 0 = no)",
    )
    
    task = Task([example, example, example, example, example], description)

    game = MyGame(
        task=task,
        signature_encoder_cls=(MyWordsSignature, MyHumanAdapter),
        signature_decoder_cls=[(MySingleChoiceSignature, MyAIAdapter)],
        signature_validator_cls=(MyWordsSignature, MyHumanAdapter),
        agent_cls=MyAgent,
    )

    ui.add_css(
        r"a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}"
    )

    with ui.footer().classes("bg-white"):
        with ui.column().classes("w-full max-w-3xl mx-auto my-4"):
            with ui.row().classes("w-full no-wrap items-center"):
                with ui.avatar().on("click", lambda: ui.navigate.to(main)):
                    ui.image(user_avatar)

                text = (
                    ui.input(placeholder="answer...")
                    .props("rounded outlined input-class=mx-3")
                    .classes("flex-grow")
                )

                async def on_send_clicked() -> None:
                    user_text = str(text.value or "").strip()
                    if not user_text:
                        return

                    # helpful normalization for decode phase
                    if game.round_state.phase == DECODING:
                        user_text = user_text.lower()

                    text.value = ""
                    post(own_id, user_avatar, user_text)

                    result = game.submit(user_text)
                    post(bot_id, bot_avatar, result["message"])

                    if result["done"]:
                        return

                    await prompt_turn()

                ui.button(
                    "Send", on_click=lambda: asyncio.create_task(on_send_clicked())
                ).props("unelevated").classes("ml-2")

            ui.markdown("notebook game logic in a chat-style NiceGUI UI").classes(
                "text-xs self-end mr-8 m-[-0.5em] text-primary"
            )

    await ui.context.client.connected()

    with ui.column().classes("w-full max-w-2xl mx-auto items-stretch pb-24"):
        chat_messages(own_id)

    ui.keyboard(
        on_key=lambda e: (
            asyncio.create_task(on_send_clicked())
            if e.key == "Enter" and (text.value or "").strip()
            else None
        )
    )

    ui.run_javascript('Array.from(document.querySelectorAll("input")).at(-1)?.focus()')

    async def prompt_turn() -> None:
        prompt_text, images = format_prompt_from_game(game)
        post(bot_id, bot_avatar, prompt_text, images=images)

        agent, _ = game.current_agent_and_description()
        use_ai = isinstance(agent.adapter, MyAIAdapter)

        if use_ai:
            done = await ai_submit_until_valid(game, own_id, bot_id, bot_avatar)
            if not done:
                await prompt_turn()

    post(bot_id, bot_avatar, "Welcome to the multimodal game framework")
    await prompt_turn()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run()
