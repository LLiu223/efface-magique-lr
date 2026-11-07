"""Shared fakes for tests/test_agent_*.py: a scripted chat model, a fake engine, a fake segmentation net.

No network, no keys. Import only from tests that need LangChain (they skip without it).
"""

import torch
from PIL import Image

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


def call(name, args=None, i=0):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": f"c{i}-{name}"}],
                     usage_metadata=dict(USAGE))


def say(text):
    return AIMessage(content=text, usage_metadata=dict(USAGE))


class ScriptedChatModel(GenericFakeChatModel):
    """Replays `messages` in order; an Exception item is raised instead. Records every input."""
    seen: list = []
    on_call: object = None

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append([str(m.content) for m in messages])
        if self.on_call:
            self.on_call(len(self.seen))
        item = next(self.messages)
        if isinstance(item, BaseException):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


def scripted(*items, on_call=None):
    return ScriptedChatModel(messages=iter(items), seen=[], on_call=on_call)


class FakeEngine:
    def __init__(self, model_name="lama"):
        self.model_name = model_name
        self.generated = []

    def set_model(self, name):
        self.model_name = name

    def generate_variations(self, image, mask, num_variations=1, **kwargs):
        self.generated.append((image, mask, num_variations, kwargs))
        return [Image.new("RGB", image.size, (0, 255, 0)) for _ in range(num_variations)]


class FakeSegNet(torch.nn.Module):
    """Logits for 21 VOC classes: `boxes` = [(class_index, x0, x1)] as fractions of the width (full height)."""

    def __init__(self, boxes):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.boxes = boxes

    def forward(self, x):
        _, _, h, w = x.shape
        out = torch.zeros(1, 21, h, w)
        out[:, 0] = 1.0
        for cls, x0, x1 in self.boxes:
            out[:, cls, h // 4: 3 * h // 4, int(x0 * w): int(x1 * w)] = 5.0
        return {"out": out}


BOTTLE, PERSON = 5, 15
