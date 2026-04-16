# %% [markdown]
# # Game prototype

# %%
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

from IPython.display import Image as IPImage, display, clear_output
from pathlib import Path
from PIL import Image as PImage
from openai import OpenAI

import base64
from io import BytesIO

import logging
import regex




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="app.log",
    filemode="a",
)

CanvasType = PImage.Image | str
QueryType = CanvasType | list[CanvasType]
AlphabetType = dict[str, QueryType] | list[str]  # first str index/name of symbol
PayloadType = list[dict[str, any]]

ENCODING = 0
DECODING = 1
VALIDATING = 2

# %%
# ───────────────────────────────────
# EXAMPLE & TASK
# ───────────────────────────────────


@dataclass
class Alphabet:
    def __init__(
        self, alphabet_name: str, indexed_letters: AlphabetType, type: int
    ):

        self.alphabet_name = alphabet_name
        self.indexed_letters = indexed_letters
        # TODO self.seperator = "" 
        self.type = type  # 0: list[str], l: dict[str, QueryType]


@dataclass
class Example:
    """
    One example puzzle: contains the message to encode,
    the allowed alphabets for encoding, and the alphabet for decoding.
    """

    def __init__(
        self,
        message: QueryType,
        encoder_alphabet: Alphabet,
        decoder_alphabet: Alphabet,
        validator_alphabet: Alphabet,
    ):
        self.message = message
        self.encoder_alphabet = encoder_alphabet
        self.decoder_alphabet = decoder_alphabet
        self.validator_alphabet = validator_alphabet


@dataclass()
class Task:
    """A set of examples forming a complete task."""

    def __init__(
        self,
        examples: list[Example],
        description: tuple[str, str, str,]
    ):
        self.examples = examples        
        self.encode_description, self.decode_description, self.validate_description = description


@dataclass()
class RoundState:
    """
    Stores the intermediate and final results of a single example:
    the cipher and the decoded message(message prime).
    Class used for logging
    """

    def __init__(
        self, example: Example
    ):
        self.example = example
        self.cipher = None
        self.ciphers = []
        self.message_prime = None
        self.message_primes = []
        self.points = 0
        self.phase = ENCODING  # 0: enc phase, 1: dec phase, 2: val phase


# ───────────────────────────────────
# Signature
# ───────────────────────────────────


class Signature(ABC):
    """Abstract base for all signatures."""

    @abstractmethod
    def check(
        self,
        query: QueryType,
        alphabet: Alphabet,
    ) -> bool:
        """Check whether the query is valid given the alphabet."""
        pass


class WordsSignature(Signature, ABC):
    """Signature allowing arbitrary combinations of symbols."""

    NAME = "" # "Valid is any word over the alphabet"
    pass


class NWordsSignature(Signature, ABC):
    """Signature restricting words to maximum length n."""

    NAME = "" # "Valid is any word over the alphabet of length n"
    pass


class SingleChoiceSignature(Signature, ABC):
    """Signature restricted to a single symbol."""

    NAME = "" # "Valid is any single choice of the alphabet"
    pass


class MultipleChoiceSignature(Signature, ABC):
    """Signature restricted to a fixed number of symbols."""

    NAME = "" # "Valid is a non repeating choice of four symbols"  # TODO. n
    pass


class PermutationsSignature(Signature, ABC):
    """Signature restricted to permutation of the entire alphabet."""

    NAME = "" # "Valid is a permutation of the alphabet"
    pass


class VariationSignature(Signature, ABC):
    """Signature allowing variation of the alphabet."""

    NAME = "" # "Valid is a variation of the alphabet"
    pass


# ───────────────────────────────────
# Adapter
# ───────────────────────────────────


class Adapter(ABC):
    """Interface for human or AI adapter interacting with the game."""

    @abstractmethod
    def present_data(
        self,
        round_state: RoundState,
        signature_name: str,
        task: str,
    ) -> PayloadType:
        """Display the query and alphabet to the adapter."""
        pass

    @abstractmethod
    def get_data(self, response: PayloadType) -> QueryType:
        """Return the adapter's response (cipher or decoded message)."""
        pass


class HumanAdapter(Adapter):
    """adapter interacting with a human user."""

    pass


class AIAdapter(Adapter):
    """adapter interacting with an AI model."""

    pass


# ───────────────────────────────────
# Agent
# ───────────────────────────────────


class Agent(ABC):
    """Handles encoding and communication with an adapter."""

    def __init__(self, signature_cls: Type[Signature], adapter_cls: Type[Adapter]):
        self.signature = signature_cls()
        self.adapter = adapter_cls()

    @abstractmethod
    def ask_adapter_for_query(self, round_state: RoundState) -> QueryType:
        """Run the adapter to produce a query."""
        pass

    @abstractmethod
    def verify_query_with_signature(
        self, query: QueryType, round_state: RoundState
    ) -> bool:
        """Check if the produced query is valid."""
        pass


# ───────────────────────────────────
# GAME
# ───────────────────────────────────


class Game(ABC):
    """Game logic handling transmission, validation, and communication."""

    def __init__(
        self,
        task: Task,
        signature_encoder_cls: tuple[Type[Signature], Type[Adapter]],
        signature_decoder_cls: list[tuple[Type[Signature], Type[Adapter]]],
        signature_validator_cls: tuple[Type[Signature], Type[Adapter]],
        agent_cls: Type[Agent],
    ):
        self.task = task

        self.agent_sender = agent_cls(*signature_encoder_cls)

        self.agent_receivers = [agent_cls(*t) for t in signature_decoder_cls]

        self.agent_validator = agent_cls(*signature_validator_cls)

        self.points = {r: 0 for r in self.agent_receivers}

        self.example_index = 0

    @abstractmethod
    def current_agent_and_description(self) -> tuple[Agent, str]:
        """Return (current_agent, current_phase_description)."""
        pass

    @abstractmethod
    def get_prompt_data(self) -> dict:
        """Return read-only data required to render the current prompt."""
        pass

    @abstractmethod
    def submit(self, query: QueryType) -> dict:
        """Validate input and advance state by one step."""
        pass

    @abstractmethod
    def step_via_agent(self) -> dict:
        """Pull input via adapter and forward it to submit()."""
        pass

# %%
# -----------------------------
# Signatures
# -----------------------------


class MyWordsSignature(WordsSignature):
    def check(self, query, alphabet) -> bool:
        query = query.strip().replace("\n", "").replace("\r", "")
        return all(c in alphabet.indexed_letters for c in regex.findall(r"\X", query))


# -----------------------------
# SingleChoice
# -----------------------------
class MySingleChoiceSignature(SingleChoiceSignature):
    def check(self, query, alphabet) -> bool:
        query = query.strip().replace("\n", "").replace("\r", "")
        return query in alphabet.indexed_letters


class MyMultipleChoiceSignature(MultipleChoiceSignature):
    def check(self, query, alphabet) -> bool:
        query = query.strip().replace("\n", "").replace("\r", "")

        return all(c in alphabet.indexed_letters for c in query) and len(query) == 4


# -----------------------------
# Adapter
# -----------------------------


class MyHumanAdapter(HumanAdapter):
    def present_data(self, round_state, signature_name, task):
        # clear_output()

        query, query_alphabet, task_alphabet = extract_data(round_state)

        print(task)

        show_query(query_alphabet, query)

        #print(f"\n{signature_name}")
        #print(f"\nAlphabet: ")

        show_alphabet(task_alphabet)

        return None

    def get_data(self, response):
        query = input("\nEnter: ")
        #print(f"\nYou typed: {query}")

        return query


class MyAIAdapter(AIAdapter):
    def present_data(self, round_state, signature_name, task):

        query, query_alphabet, task_alphabet = extract_data(round_state)

        if isinstance(task_alphabet.indexed_letters, list) and isinstance(
            query_alphabet.indexed_letters, list
        ):
            task = "\n".join(
                [
                    task,
                    f"\n{signature_name}\nQuery: {query}\nAlphabet: {task_alphabet.indexed_letters}",
                    "Just return the answer, nothing else",
                ]
            )
            #print(task)
            payload = build_multi_modal_payload(final_text=task)

        elif isinstance(query_alphabet.indexed_letters, dict):
            task = "\n".join(
                [
                    task,
                    f"\n{signature_name}\nAlphabet: {task_alphabet.indexed_letters}",
                    "Just return the answer, nothing else",
                ]
            )
            #print(task)
            payload = build_multi_modal_payload(
                image_text_pairs=[
                    (query_alphabet.indexed_letters[k], k) for k in list(query)
                ],
                final_text=task,
            )

        elif isinstance(task_alphabet.indexed_letters, dict):
            task = "\n".join(
                [
                    task,
                    f"\n{signature_name}\nQuery: {query}\nAlphabet are the image index pairs",
                    "Just return the answer as string of indices, nothing else",
                ]
            )
            payload = build_multi_modal_payload(
                image_text_pairs=[
                    (task_alphabet.indexed_letters[k], k)
                    for k in task_alphabet.indexed_letters
                ],
                final_text=task,
            )

        response = send_request(payload)

        return response

    def get_data(
        self, response
    ):

        query = get_message_text(response)
        #print(f"\nYou typed: {query}")
        return query


# -----------------------------
# Agent
# -----------------------------


class MyAgent(Agent):
    def __init__(self, signature_cls, adapter_cls):
        super().__init__(signature_cls, adapter_cls)

    def ask_adapter_for_query(self, round_state, description):
        response = self.adapter.present_data(
            round_state, type(self.signature).NAME, description
        )
        return self.adapter.get_data(response)

      # Not query from round state, because first query validated then written to round_state
    def verify_query_with_signature(self, query, round_state):
        _, _, task_alphabet = extract_data(round_state)
        return self.signature.check(query, task_alphabet)


# -----------------------------
# Game
# -----------------------------


class MyGame(Game):
    def __init__(
        self,
        task,
        signature_encoder_cls,
        signature_decoder_cls,
        signature_validator_cls,
        agent_cls,
    ):
        super().__init__(
            task,
            signature_encoder_cls,
            signature_decoder_cls,
            signature_validator_cls,
            agent_cls,
        )

        # For simplicity, assume only one receiver
        self.agent_receiver = self.agent_receivers[0]
        self.round_state = RoundState(self.task.examples[0])


    def current_agent_and_description(self):
        
        if self.round_state.phase == ENCODING:
            return self.agent_sender, self.task.encode_description
        
        if self.round_state.phase == DECODING:
            return self.agent_receiver, self.task.decode_description
        
        return self.agent_validator, self.task.validate_description


    def get_prompt_data(self):
        agent, desc = self.current_agent_and_description()
        query, query_alphabet, task_alphabet = extract_data(self.round_state)

        return {
            "phase": self.round_state.phase,
            "description": desc,
            "signature_name": type(agent.signature).NAME,
            "query": query,
            "query_alphabet": query_alphabet,
            "task_alphabet": task_alphabet,
        }


    def submit(self, query: str):
        agent, _ = self.current_agent_and_description()

        ok = agent.verify_query_with_signature(query, self.round_state)
        if not ok:
            return {"accepted": False, "message": "Invalid for this signature/alphabet.", "done": False}

        # phase ENCODING -> DECODING
        if self.round_state.phase == ENCODING:
            self.round_state.cipher = query
            self.round_state.ciphers.append(query)
            self.round_state.phase = 1
            return {"accepted": True, "message": "Cipher accepted.", "done": False}

        # phase DECODING -> VALIDATING
        if self.round_state.phase == DECODING:
            self.round_state.message_prime = query
            self.round_state.message_primes.append(query)
            self.round_state.phase = 2
            return {"accepted": True, "message": "Decoded message accepted.", "done": False}

        # phase VALIDATING -> next example / done
        self.round_state.points = query
        self.points[self.agent_receiver] += int(query)

        self.example_index += 1
        if self.example_index < len(self.task.examples):
            self.round_state = RoundState(self.task.examples[self.example_index])
            return {"accepted": True, "message": "Next example.", "done": False}

        return {"accepted": True, "message": "Task complete.", "done": True}
    
    def step_via_agent(self):
        
        agent, desc = self.current_agent_and_description()
        query = agent.ask_adapter_for_query(self.round_state, desc)
        return self.submit(query)

# %%
if __name__ == "__main__":
    emoji_alphabet = Alphabet("emojis", ["❄️", "🛳️", "❤️", "💥", "🌊", "🧊", "💀", "🦈", "🚢", "🌹"], 0)
    validator_alphabet = Alphabet("Points", ["0", "1"], 0)
    letter_alphabet = Alphabet("lower_letters", list("abcdefghijklmnopqrstuvwxyz"), 0)
    country_alphabet = Alphabet("countries", ["USA", "Germany", "France"], 0)
    image_alphabet = Alphabet("images", {"0": load_local_image(Path("images/logoA.jpg")), "1": load_local_image(Path("images/logoB.jpg")), "2": load_local_image(Path("images/titanic.jpg")), "3": load_local_image(Path("images/usa_1.jpg"))
                                             , "4": load_local_image(Path("images/usa_2.jpg")), "5": load_local_image(Path("images/usa_3.jpg")), "6": load_local_image(Path("images/usa_4.jpg"))}, 1)

    description = ("Choose an emoji combination that describes the movie the best", "Which movie is describes by the emoji's", "Is this the movie you had in mind?")

    examples = [
        Example("Titanic", emoji_alphabet, letter_alphabet, validator_alphabet),
        # Example("Inception", emoji_alphabet, letter_alphabet, validator_alphabet),
        # Example("Avatar", emoji_alphabet, letter_alphabet, validator_alphabet),
    ]
    task = Task(examples, description)

#--------------------------------------------------------------------------------------------------------------------------#

    description_4i1w = ("Choose 4 images which represent the country the best", "Choose the country best described by the four images", "Is this the country you had in mind?")
    examples_4i1w = [
        Example("USA", image_alphabet, country_alphabet, validator_alphabet),
    ]
    task_4i1w = Task(examples_4i1w, description_4i1w)

#--------------------------------------------------------------------------------------------------------------------------#

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
    
    task = Task([example], description)

#--------------------------------------------------------------------------------------------------------------------------#

    game = MyGame(
        task=task,
        signature_encoder_cls=(MyWordsSignature, MyHumanAdapter),
        signature_decoder_cls=[(MySingleChoiceSignature, MyAIAdapter)],
        signature_validator_cls=(MyWordsSignature, MyHumanAdapter),
        agent_cls=MyAgent,
    )

    while True:
        
        result = game.step_via_agent()

        print(result["message"])

        if result["done"]:
            break

# %%
### chatbot methods


def load_local_image(path: Path) -> PImage.Image:
    """
    Load an image from disk as a PIL Image.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    return PImage.open(path)
    
def show_alphabet(alphabet: Alphabet):
    """
    For each entry in the image alphabet, print the index
    and then display the corresponding image in the notebook.
    """
    if alphabet.type == 0:  # list[str] alphabet

        print(f"{alphabet.indexed_letters}")

    else: # dict[] alphabet

        for idx, pil_img in alphabet.indexed_letters.items():
            print(idx)
            display(pil_img)

def show_query(alphabet: Alphabet, query: QueryType):
    """
    For a sequence like "0031", print the index and display
    the corresponding image for each character.
    """

    if alphabet.type == 0:  # list[str] alphabet

        print(f"\nQuery: {query}")
        
    else:  # dict[] alphabet

        for index in query:
            if index not in alphabet.indexed_letters:
                raise KeyError(f"Index '{index}' not found in alphabet.")
            pil_img = alphabet.indexed_letters[index]
            print(index)
            display(pil_img)




def encode_image_base64(image: PImage.Image) -> str:
    """
    Encode a PIL Image as base64 JPEG.
    """
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_message_text(response) -> str:
    print(response)
    return response.choices[0].message.content or ""


def send_request(payload: PayloadType) -> str:
    """
    Send a single multimodal payload to the OpenAI-compatible chat API
    and return only the assistant's textual response.
    """

    client = OpenAI(
        base_url="https://llm.srv.webis.de/openai/v1/",  # "https://chat.web.webis.de/openai/" - with key or "https://llm.srv.webis.de/openai/v1/" in VPN
        api_key="sk-",
    )

    response = client.chat.completions.create(
        model="gemma3-4b",
        messages=[
            {
                "role": "user",
                "content": payload,
            }
        ],
        max_tokens=300,
    )

    return response


def build_multi_modal_payload(
    final_text: str,
    image_text_pairs: list[tuple[PImage.Image, str]] = [],
) -> PayloadType:
    """
    Build a multimodal payload from optional image-text pairs and a final instruction.
    """

    payload  = []

    # Add image-text pairs if any
    for image, text in image_text_pairs:
        payload.append({"type": "text", "text": text})
        payload.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_base64(image)}"
                },
            }
        )

    # Always add the final text
    #final_text = "To which image belongs the following caption: \"bill\" "
    payload.append({"type": "text", "text": final_text})

    return payload

# %%
# Helper for RoundState

def extract_data(round_state: RoundState) -> tuple[QueryType, Alphabet, Alphabet]:
    """
    Given the current state we extract the query, over the query_alphabet, needed in the current state, as well as the task_alphabet over which the answer to the task should be encoded
    """

    phase = round_state.phase
    if phase == 0:
        query, query_alphabet, task_alphabet = (
            round_state.example.message,
            round_state.example.decoder_alphabet,
            round_state.example.encoder_alphabet,
        )
    elif phase == 1:
        query, query_alphabet, task_alphabet = (
            round_state.cipher,
            round_state.example.encoder_alphabet,
            round_state.example.decoder_alphabet,
        )
    else:
        query, query_alphabet, task_alphabet = (
            round_state.message_prime,
            round_state.example.decoder_alphabet,
            round_state.example.validator_alphabet,
        )

    return query, query_alphabet, task_alphabet


