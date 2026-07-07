"""
A2UI (Agent-to-User Interface) support for the A2A server.

Implements the A2UI v0.9 A2A extension so agents can return rich,
declarative UI instead of plain text. Adapted from the official A2UI
Python agent SDK (https://github.com/a2ui-project/a2ui, Apache 2.0),
ported to the protobuf-based a2a-sdk >= 1.0 types used by this server.

Wire format (A2UI over A2A):
  - The agent card advertises an AgentExtension with URI
    "https://a2ui.org/a2a-extension/a2ui/v<version>".
  - A client opts in per-request by listing that URI in its requested
    extensions (X-A2A-Extensions header) or on the message itself.
  - Each A2UI message is carried as an A2A Part whose `data` field holds
    the A2UI JSON envelope and whose media type is "application/json+a2ui"
    (v0.8/v0.9) or "application/a2ui+json" (v1.0+).
  - User interactions come back as Parts with the same media type holding
    a client-to-server `action` payload.

LLM contract: the model emits normal conversational text interleaved with
A2UI JSON blocks wrapped in <a2ui-json> ... </a2ui-json> tags (the same
delimiters the official SDK parsers use), which this module splits into
text and data parts.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from a2a.types import AgentCard, AgentExtension, Message, Part
from a2a.helpers.proto_helpers import new_text_part
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict, ParseDict

from oai_agent_core.utils.a2ui_prompt import (
    A2UI_CLOSE_TAG,
    A2UI_OPEN_TAG,
    BASIC_CATALOG_ID,
    build_agui_instructions,
    get_a2ui_system_instructions,
    is_agui_enabled,
    resolve_catalog,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants (mirroring the official A2UI SDK)
# ---------------------------------------------------------------------------

A2UI_EXTENSION_BASE_URI = "https://a2ui.org/a2a-extension/a2ui"
A2UI_DEFAULT_VERSION = "0.9"

A2UI_MIME_TYPE = "application/a2ui+json"          # v1.0+
DEPRECATED_A2UI_MIME_TYPE = "application/json+a2ui"  # v0.8 / v0.9
MIME_TYPE_KEY = "mimeType"

AGENT_EXTENSION_SUPPORTED_CATALOG_IDS_KEY = "supportedCatalogIds"
AGENT_EXTENSION_ACCEPTS_INLINE_CATALOGS_KEY = "acceptsInlineCatalogs"

# Message-metadata keys sent by A2UI clients
A2UI_CLIENT_CAPABILITIES_KEY = "a2uiClientCapabilities"
A2UI_CLIENT_DATA_MODEL_KEY = "a2uiClientDataModel"


def a2ui_mime_type(version: Optional[str]) -> str:
    """Returns the A2UI part media type for a protocol version."""
    if version is None or version.lstrip("v") in ("0.8", "0.9", "0.9.1"):
        return DEPRECATED_A2UI_MIME_TYPE
    return A2UI_MIME_TYPE


# ---------------------------------------------------------------------------
# Agent card extension + per-request negotiation
# ---------------------------------------------------------------------------

def get_a2ui_agent_extension(
    version: str = A2UI_DEFAULT_VERSION,
    supported_catalog_ids: Optional[List[str]] = None,
    accepts_inline_catalogs: bool = False,
) -> AgentExtension:
    """Builds the AgentExtension entry advertising A2UI support."""
    params: Dict[str, Any] = {}
    if accepts_inline_catalogs:
        params[AGENT_EXTENSION_ACCEPTS_INLINE_CATALOGS_KEY] = True
    if supported_catalog_ids is None:
        supported_catalog_ids = [BASIC_CATALOG_ID]
    if supported_catalog_ids:
        params[AGENT_EXTENSION_SUPPORTED_CATALOG_IDS_KEY] = supported_catalog_ids

    extension = AgentExtension(
        uri=f"{A2UI_EXTENSION_BASE_URI}/v{version}",
        description="Provides agent driven UI using the A2UI JSON format.",
    )
    if params:
        extension.params.CopyFrom(ParseDict(params, struct_pb2.Struct()))
    return extension


def _version_key(uri: str) -> Tuple[int, ...]:
    version_str = uri.replace(f"{A2UI_EXTENSION_BASE_URI}/v", "")
    try:
        return tuple(int(p) for p in version_str.split("."))
    except ValueError:
        return (0,)


def _agent_a2ui_extensions(agent_card: Optional[AgentCard]) -> List[str]:
    if agent_card is None:
        return []
    try:
        extensions = agent_card.capabilities.extensions
    except AttributeError:
        return []
    return [
        ext.uri
        for ext in extensions
        if ext.uri and ext.uri.startswith(A2UI_EXTENSION_BASE_URI)
    ]


def _requested_a2ui_extensions(context: Any) -> List[str]:
    requested: List[str] = []
    try:
        requested.extend(
            ext
            for ext in (context.requested_extensions or set())
            if isinstance(ext, str) and ext.startswith(A2UI_EXTENSION_BASE_URI)
        )
    except AttributeError:
        pass

    message = getattr(context, "message", None)
    if message is not None:
        try:
            requested.extend(
                ext
                for ext in message.extensions
                if isinstance(ext, str) and ext.startswith(A2UI_EXTENSION_BASE_URI)
            )
        except AttributeError:
            pass
    return requested


def try_activate_a2ui_extension(
    context: Any, agent_card: Optional[AgentCard]
) -> Optional[str]:
    """
    Negotiates the A2UI extension for one request.

    Returns the protocol version string (e.g. "0.9") when the client requested
    an A2UI extension version this agent advertises, otherwise None (plain-text
    behaviour).
    """
    requested = _requested_a2ui_extensions(context)
    if not requested:
        return None

    advertised = _agent_a2ui_extensions(agent_card)
    matched = [uri for uri in requested if uri in advertised]
    if not matched:
        return None

    selected = max(matched, key=_version_key)
    return selected.replace(f"{A2UI_EXTENSION_BASE_URI}/v", "")


# ---------------------------------------------------------------------------
# A2A Part helpers
# ---------------------------------------------------------------------------

def create_a2ui_part(a2ui_data: Dict[str, Any], version: Optional[str] = None) -> Part:
    """Wraps one A2UI JSON message in an A2A data Part."""
    mime = a2ui_mime_type(version)
    part = Part(
        data=ParseDict(a2ui_data, struct_pb2.Value()),
        media_type=mime,
    )
    # Some renderers look for the MIME type in the part metadata instead of
    # the media_type field, so set both.
    part.metadata.CopyFrom(ParseDict({MIME_TYPE_KEY: mime}, struct_pb2.Struct()))
    return part


def is_a2ui_part(part: Part) -> bool:
    """True when an A2A Part carries an A2UI payload (either MIME marker)."""
    known = (A2UI_MIME_TYPE, DEPRECATED_A2UI_MIME_TYPE)
    if getattr(part, "media_type", "") in known:
        return True
    try:
        metadata = MessageToDict(part.metadata)
    except Exception:
        return False
    return metadata.get(MIME_TYPE_KEY) in known


def get_a2ui_data(part: Part) -> Optional[Any]:
    """Extracts the A2UI JSON payload from an A2A Part, or None."""
    if not is_a2ui_part(part):
        return None
    try:
        return MessageToDict(part.data)
    except Exception as exc:
        logger.warning("Failed to decode A2UI part data: %s", exc)
        return None


def get_a2ui_message_metadata(message: Optional[Message]) -> Dict[str, Any]:
    """Returns the a2ui* keys (client capabilities, data model) from message metadata."""
    if message is None:
        return {}
    try:
        metadata = MessageToDict(message.metadata)
    except Exception:
        return {}
    return {k: v for k, v in metadata.items() if k.startswith("a2ui")}


# ---------------------------------------------------------------------------
# LLM response parsing: <a2ui-json> ... </a2ui-json> blocks
# ---------------------------------------------------------------------------

_A2UI_BLOCK_PATTERN = re.compile(
    f"{re.escape(A2UI_OPEN_TAG)}(.*?){re.escape(A2UI_CLOSE_TAG)}", re.DOTALL
)
_TRAILING_COMMA_PATTERN = re.compile(r",\s*([\]}])")

# LLMs drift on exact protocol strings despite prompt rules; normalize the
# known failure modes before emitting (same idea as the official SDK's
# payload_fixer). Renderers hard-fail on an unknown catalogId.
_BASIC_CATALOG_DRIFT = re.compile(
    r"^https://a2ui\.org/specification/v0[._]9(?:[._]\d+)?/catalogs/basic/catalog\.json$"
)
_KEBAB_VALUE = re.compile(r"-([a-z])")
# Component properties whose values are camelCase enums the model tends to
# write in CSS kebab-case (space-between, scale-down, ...).
_ENUM_PROPERTY_KEYS = ("justify", "align", "fit", "direction", "axis", "displayStyle")


def normalize_a2ui_message(message: Any) -> Any:
    """Fixes known LLM drift in one A2UI message (in place)."""
    if not isinstance(message, dict):
        return message

    create_surface = message.get("createSurface")
    if isinstance(create_surface, dict):
        catalog_id = create_surface.get("catalogId")
        if isinstance(catalog_id, str) and _BASIC_CATALOG_DRIFT.match(catalog_id):
            create_surface["catalogId"] = BASIC_CATALOG_ID

    update_components = message.get("updateComponents")
    if isinstance(update_components, dict):
        for component in update_components.get("components") or []:
            if not isinstance(component, dict):
                continue
            for key in _ENUM_PROPERTY_KEYS:
                value = component.get(key)
                if isinstance(value, str) and "-" in value:
                    component[key] = _KEBAB_VALUE.sub(
                        lambda m: m.group(1).upper(), value
                    )
    return message


def has_a2ui_content(content: str) -> bool:
    return A2UI_OPEN_TAG in content


def _parse_json_block(json_string: str) -> Optional[Any]:
    """
    Parses one delimited block, tolerating markdown fences and trailing
    commas, and normalizing known LLM drift in the parsed messages.
    """
    cleaned = json_string.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")]
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            data = json.loads(_TRAILING_COMMA_PATTERN.sub(r"\1", cleaned))
        except json.JSONDecodeError as exc:
            logger.warning("Discarding malformed A2UI JSON block: %s", exc)
            return None
    if isinstance(data, list):
        return [normalize_a2ui_message(msg) for msg in data]
    return normalize_a2ui_message(data)


def split_a2ui_content(content: str) -> List[Tuple[str, Any]]:
    """
    Splits raw LLM output into ("text", str) and ("a2ui", dict) segments.
    A2UI blocks holding a JSON array are flattened to one segment per message.
    """
    segments: List[Tuple[str, Any]] = []
    last_end = 0
    for match in _A2UI_BLOCK_PATTERN.finditer(content):
        text = content[last_end: match.start()].strip()
        if text:
            segments.append(("text", text))
        data = _parse_json_block(match.group(1))
        if isinstance(data, list):
            segments.extend(("a2ui", msg) for msg in data if isinstance(msg, dict))
        elif isinstance(data, dict):
            segments.append(("a2ui", data))
        last_end = match.end()
    trailing = content[last_end:].strip()
    if trailing:
        segments.append(("text", trailing))
    return segments


def build_a2ui_parts(
    content: str,
    version: Optional[str] = None,
    fallback_text: Optional[str] = None,
) -> List[Part]:
    """Converts raw LLM output into a list of A2A text/data Parts."""
    parts: List[Part] = []
    for kind, value in split_a2ui_content(content):
        if kind == "text":
            parts.append(new_text_part(value))
        else:
            parts.append(create_a2ui_part(value, version=version))
    if not parts and fallback_text:
        parts.append(new_text_part(fallback_text))
    return parts


def extract_plain_text(content: str) -> str:
    """Returns only the conversational text, with A2UI blocks removed."""
    return " ".join(
        value for kind, value in split_a2ui_content(content) if kind == "text"
    ).strip()


class A2uiStreamSegmenter:
    """
    Incremental splitter for streamed LLM deltas.

    feed() returns completed ("text", str) / ("a2ui", dict) segments as soon
    as they are available: text outside tags flows through eagerly (holding
    back only a possible partial open tag), while A2UI JSON is buffered until
    its closing tag arrives.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_block = False

    @staticmethod
    def _partial_suffix_len(text: str, tag: str) -> int:
        max_len = min(len(text), len(tag) - 1)
        for length in range(max_len, 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0

    def feed(self, delta: str) -> List[Tuple[str, Any]]:
        self._buffer += delta
        segments: List[Tuple[str, Any]] = []
        while True:
            if self._in_block:
                idx = self._buffer.find(A2UI_CLOSE_TAG)
                if idx < 0:
                    break
                block = self._buffer[:idx]
                self._buffer = self._buffer[idx + len(A2UI_CLOSE_TAG):]
                self._in_block = False
                data = _parse_json_block(block)
                if isinstance(data, list):
                    segments.extend(
                        ("a2ui", msg) for msg in data if isinstance(msg, dict)
                    )
                elif isinstance(data, dict):
                    segments.append(("a2ui", data))
            else:
                idx = self._buffer.find(A2UI_OPEN_TAG)
                if idx >= 0:
                    text = self._buffer[:idx]
                    if text.strip():
                        segments.append(("text", text))
                    self._buffer = self._buffer[idx + len(A2UI_OPEN_TAG):]
                    self._in_block = True
                    continue
                hold = self._partial_suffix_len(self._buffer, A2UI_OPEN_TAG)
                emit = self._buffer[: len(self._buffer) - hold]
                self._buffer = self._buffer[len(self._buffer) - hold:]
                if emit.strip():
                    segments.append(("text", emit))
                break
        return segments

    def flush(self) -> List[Tuple[str, Any]]:
        """Returns whatever remains; an unterminated A2UI block degrades to text."""
        segments: List[Tuple[str, Any]] = []
        if self._buffer.strip():
            if self._in_block:
                data = _parse_json_block(self._buffer)
                if isinstance(data, list):
                    segments.extend(
                        ("a2ui", msg) for msg in data if isinstance(msg, dict)
                    )
                elif isinstance(data, dict):
                    segments.append(("a2ui", data))
                else:
                    segments.append(("text", self._buffer))
            else:
                segments.append(("text", self._buffer))
        self._buffer = ""
        self._in_block = False
        return segments


# ---------------------------------------------------------------------------
# Inbound user actions
# ---------------------------------------------------------------------------

def format_user_action(event: Dict[str, Any]) -> str:
    """Renders a client-to-server A2UI event as text the LLM can act on."""
    action = event.get("action")
    if isinstance(action, dict):
        return (
            f"The user triggered the UI action '{action.get('name', 'unknown')}' "
            f"(surface '{action.get('surfaceId', '')}', "
            f"component '{action.get('sourceComponentId', '')}') "
            f"with values: {json.dumps(action.get('context', {}), ensure_ascii=False)}"
        )
    error = event.get("error")
    if isinstance(error, dict):
        return f"The client reported a UI error: {json.dumps(error, ensure_ascii=False)}"
    return f"A2UI client event: {json.dumps(event, ensure_ascii=False)}"


# ---------------------------------------------------------------------------
# Echo stripping
# ---------------------------------------------------------------------------
# The A2UI instructions are injected into the model input by
# BaseAgent._augment_message (oai_agent_core.utils.a2ui_prompt) when
# enabled_agui_protocol is set. Some framework streams (e.g. LangGraph with
# stream_mode="values") echo the input message back as the first chunk, so
# the injected block must be removed from outbound text — both for a clean
# transcript and because the instructions contain an example A2UI block that
# would otherwise be parsed and rendered as real UI.

def strip_a2ui_instructions(text: str, instructions: Optional[str] = None) -> str:
    """
    Removes any echo of the injected A2UI instruction block from text.

    Args:
        text: The model output to clean.
        instructions: The exact block that was injected for this agent
            (from build_agui_instructions). Falls back to the default
            basic-catalog block when omitted.
    """
    if not text:
        return text
    return text.replace(instructions or get_a2ui_system_instructions(), "")
