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


# --- precise selection (job 023): Grounding DINO, SAM 2, ViTMatte -------------------------------------

def dark(img):
    """Pixels darker than mid-grey: the test images draw objects in black on white."""
    import numpy as np
    return np.asarray(img.convert("L")) < 128


class _Param(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))


class FakeGdinoProcessor:
    """`boxes` = [(label, x0, y0, x1, y1, score)] as fractions of the image; records texts and thresholds."""

    def __init__(self, boxes=()):
        self.boxes, self.texts, self.thresholds = list(boxes), [], []

    def __call__(self, images, text, return_tensors=None):
        self.texts.append(text)
        return {"input_ids": torch.zeros(1, 4, dtype=torch.long), "pixel_values": torch.zeros(1, 3, 8, 8)}

    def post_process_grounded_object_detection(self, outputs, input_ids, threshold, text_threshold, target_sizes):
        self.thresholds.append((threshold, text_threshold))
        h, w = target_sizes[0]
        keep = [b for b in self.boxes if b[5] >= threshold]
        return [{"boxes": torch.tensor([[b[1] * w, b[2] * h, b[3] * w, b[4] * h] for b in keep]).reshape(-1, 4),
                 "scores": torch.tensor([b[5] for b in keep]), "text_labels": [b[0] for b in keep]}]


class FakeGdinoModel(_Param):
    def forward(self, **kwargs):
        return None


class FakeSamProcessor:
    """Segments `segment(crop) -> bool mask` inside the prompt box; by default the dark pixels opened by 5x5,
    so 1-px strands (hair) are missed like a real SAM edge. Records (crop size, box, points) per call."""

    def __init__(self, segment=None):
        self.segment, self.calls = segment, []

    def __call__(self, images, input_boxes, input_points=None, input_labels=None, return_tensors=None):
        import cv2
        import numpy as np
        x0, y0, x1, y1 = input_boxes[0][0]
        self.calls.append((images.size, (x0, y0, x1, y1), input_points[0][0] if input_points else []))
        if self.segment is not None:
            m = np.asarray(self.segment(images), bool)
        else:
            m = cv2.morphologyEx(dark(images).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        inside = np.zeros_like(m)
        inside[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True
        return {"pixel_values": torch.tensor(m & inside, dtype=torch.float32)[None, None, None],
                "original_sizes": torch.tensor([[images.height, images.width]])}

    def post_process_masks(self, masks, original_sizes):
        return [masks[0] > 0.5]


class FakeSamModel(_Param):
    def forward(self, pixel_values=None, multimask_output=True, **kwargs):
        return type("Out", (), {"pred_masks": pixel_values})()


class FakeVitMatteProcessor:
    """Alpha = darkness of the crop, padded to /32 like the real processor. Records the trimaps."""

    def __init__(self):
        self.trimaps = []

    def __call__(self, images, trimaps, return_tensors=None):
        import numpy as np
        self.trimaps.append(np.asarray(trimaps))
        a = torch.tensor(dark(images), dtype=torch.float32)
        h, w = a.shape
        out = torch.zeros(1, 4, -(-h // 32) * 32, -(-w // 32) * 32)
        out[0, 0, :h, :w] = a
        return {"pixel_values": out}


class FakeVitMatteModel(_Param):
    def forward(self, pixel_values):
        return type("Out", (), {"alphas": pixel_values[:, :1]})()
