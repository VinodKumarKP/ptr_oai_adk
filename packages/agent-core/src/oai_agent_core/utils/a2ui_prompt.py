"""
A2UI (Agent-to-User Interface) prompt instructions.

Single source of truth for the LLM instructions that make an agent emit
A2UI v0.9 JSON (https://github.com/a2ui-project/a2ui) alongside its
conversational text. Injection happens in BaseAgent._augment_message, so it
works identically for every framework backend (LangGraph, Strands, CrewAI,
OpenAI, Anthropic).

Configuration lives under crew_config.agui_config in the agent config:

    crew_config:
      agui_config:
        enabled: true
        # Optional — omit both to use the standard A2UI basic catalog:
        catalog_path: ./catalogs/my_catalog.json   # A2UI catalog schema file
        catalog_id: mycompany.com:my-catalog       # overrides the file's $id

When catalog_path is set, the component reference in the system prompt is
generated from the catalog schema, so the model only uses components the
client actually registered.

The A2A server layer (oai_agent_server.a2a.a2ui_support) imports these
helpers to parse the delimited blocks out of the model output and to strip
any echo of the instructions from streamed transcripts.
"""

import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CREW_CONFIG_KEY = "crew_config"
AGUI_CONFIG_KEY = "agui_config"
AGUI_ENABLED_KEY = "enabled"
AGUI_CATALOG_PATH_KEY = "catalog_path"
AGUI_CATALOG_ID_KEY = "catalog_id"

A2UI_OPEN_TAG = "<a2ui-json>"
A2UI_CLOSE_TAG = "</a2ui-json>"

BASIC_CATALOG_ID = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"

_BASIC_COMPONENTS_SECTION = """\
- Layout: `Row` / `Column` {children, justify?, align?}, `List` {children, direction?},
  `Card` {child}, `Tabs` {tabs: [{title, child}]}, `Modal` {trigger, content}, `Divider` {axis?}
- Display: `Text` {text, variant?: h1|h2|h3|h4|h5|caption|body},
  `Image` {url, fit?, variant?}, `Icon` {name}, `Video` {url}, `AudioPlayer` {url}
- Input: `Button` {child, variant?: default|primary|borderless, action},
  `TextField` {label, value?, variant?: shortText|longText|number|obscured, validationRegexp?},
  `CheckBox` {label, value}, `ChoicePicker` {label?, options, value, variant?},
  `Slider` {value, min?, max, label?}, `DateTimeInput` {value, enableDate?, enableTime?, label?}"""


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_agui_config(agent_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Returns crew_config.agui_config, or {} when absent/malformed."""
    if not isinstance(agent_config, dict):
        return {}
    crew_config = agent_config.get(CREW_CONFIG_KEY)
    if not isinstance(crew_config, dict):
        return {}
    agui_config = crew_config.get(AGUI_CONFIG_KEY)
    return agui_config if isinstance(agui_config, dict) else {}


def is_agui_enabled(agent_config: Optional[Dict[str, Any]]) -> bool:
    """True when the agent opts into emitting A2UI."""
    return bool(get_agui_config(agent_config).get(AGUI_ENABLED_KEY))


# ---------------------------------------------------------------------------
# Catalog loading and prompt generation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _load_catalog(resolved_path: str) -> Dict[str, Any]:
    with open(resolved_path, encoding="utf-8") as fh:
        return json.load(fh)


def load_catalog(path: str) -> Dict[str, Any]:
    """Loads (and caches) an A2UI catalog schema from a JSON file."""
    return _load_catalog(os.path.abspath(path))


def _property_summary(name: str, schema: Any, required: List[str]) -> str:
    suffix = "" if name in required else "?"
    if isinstance(schema, dict):
        enum = schema.get("enum")
        if isinstance(enum, list) and 0 < len(enum) <= 8:
            return f"{name}{suffix}: {'|'.join(str(v) for v in enum)}"
    return f"{name}{suffix}"


def _component_properties(definition: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Finds the property block of one component definition. Supports both the
    official catalog layout (allOf list with a `component: {const: ...}`
    entry) and a plain `{properties: {...}}` object.
    """
    candidates = definition.get("allOf") if isinstance(definition.get("allOf"), list) else [definition]
    for entry in candidates:
        if isinstance(entry, dict) and isinstance(entry.get("properties"), dict):
            if "component" in entry["properties"] or entry is definition:
                return entry
    return None


def generate_components_section(catalog: Dict[str, Any]) -> str:
    """
    Builds the component-reference block of the system prompt from an A2UI
    catalog schema: one line per component with its properties, `?` marking
    optional ones and small enums inlined.
    """
    components = catalog.get("components")
    if not isinstance(components, dict) or not components:
        raise ValueError("Catalog has no 'components' map to generate a prompt from")

    lines = []
    for name, definition in components.items():
        block = _component_properties(definition) if isinstance(definition, dict) else None
        if block is None:
            lines.append(f"- `{name}` {{}}")
            continue
        required = block.get("required", [])
        props = [
            _property_summary(prop, schema, required)
            for prop, schema in block["properties"].items()
            if prop != "component"
        ]
        lines.append(f"- `{name}` {{{', '.join(props)}}}")
    return "\n".join(lines)


def resolve_catalog(agent_config: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Resolves the catalog id and components section for an agent config.
    Falls back to the A2UI basic catalog when no custom catalog is set or
    the custom catalog fails to load.
    """
    agui_config = get_agui_config(agent_config)
    catalog_path = agui_config.get(AGUI_CATALOG_PATH_KEY)
    catalog_id = agui_config.get(AGUI_CATALOG_ID_KEY)

    if catalog_path:
        try:
            catalog = load_catalog(str(catalog_path))
            return {
                "catalog_id": str(catalog_id or catalog.get("$id") or catalog_path),
                "components_section": generate_components_section(catalog),
            }
        except Exception as exc:
            logger.error(
                "Failed to load A2UI catalog '%s' (%s); falling back to basic catalog",
                catalog_path, exc,
            )

    return {
        "catalog_id": str(catalog_id or BASIC_CATALOG_ID),
        "components_section": _BASIC_COMPONENTS_SECTION,
    }


# ---------------------------------------------------------------------------
# System instructions
# ---------------------------------------------------------------------------

def get_a2ui_system_instructions(
    catalog_id: str = BASIC_CATALOG_ID,
    components_section: str = _BASIC_COMPONENTS_SECTION,
    client_capabilities: Optional[Dict[str, Any]] = None,
) -> str:
    """Returns the A2UI instruction block to append to the model input."""
    custom_catalog_note = ""
    if catalog_id != BASIC_CATALOG_ID:
        custom_catalog_note = (
            "\nThe examples below use generic component names for illustration only —"
            "\nyour response MUST use ONLY the components listed above.\n"
        )

    instructions = f"""
## A2UI OUTPUT FORMAT (required for this request)

The client rendering your response supports A2UI v0.9, so respond with an
interactive UI instead of long prose whenever it helps (forms, lists, cards,
confirmations). Follow these rules exactly:

- Wrap every A2UI JSON payload in `{A2UI_OPEN_TAG}` and `{A2UI_CLOSE_TAG}` tags.
  You may put short conversational text before/after the blocks. Do not use
  markdown code fences inside the tags.
- The payload MUST be a JSON ARRAY of A2UI v0.9 messages. Every message object
  MUST include `"version": "v0.9"` plus exactly one of:
  - `"createSurface"`: {{"surfaceId": <unique id>, "catalogId": "{catalog_id}", "sendDataModel": true}}
    Always send this first for a new surface. Never reuse an existing surfaceId.
  - `"updateComponents"`: {{"surfaceId": ..., "components": [ ... ]}}
  - `"updateDataModel"`: {{"surfaceId": ..., "path": "/some/path", "value": <json>}}
    Always include an explicit `path` (use "/" for the whole model).
  - `"deleteSurface"`: {{"surfaceId": ...}}

### Components (flat adjacency list)
`components` is a FLAT array; components refer to children by string id, never
inline. The component with `"id": "root"` MUST be first, and parents MUST
appear before their children. Every component has `id` and `component` plus
type-specific properties (`?` marks optional):

{components_section}

- `children` is an array of ids, or a template {{"componentId": <id>, "path": "/list/path"}}
  to repeat one component over a data-model array.

### Data binding and actions
- Bind any dynamic value with {{"path": ...}} instead of a literal.
  Bind every input component's `value` to a data-model path.
- Path scoping (IMPORTANT):
  - Components OUTSIDE a template: absolute paths starting with "/", e.g. {{"path": "/username"}}.
  - Components INSIDE a `children` template (repeated per list item): RELATIVE
    paths with NO leading slash, resolved against the current item — e.g.
    {{"path": "name"}} for /hotels/N/name. An absolute path inside a template
    resolves against the root and will render empty.
- Buttons and interactive components use
  `"action": {{"event": {{"name": <action_name>, "context": {{<key>: {{"path": ...}} | <literal>}}}}}}`.
  The same path scoping applies to context values. The event is sent back to
  you with the resolved context values when the user interacts.
- Copy `catalogId` EXACTLY as given above. Component names, property names and
  enum values are exact and case-sensitive (e.g. `spaceBetween`, never `space-between`).
{custom_catalog_note}
### Example (login form)
{A2UI_OPEN_TAG}
[
  {{"version": "v0.9", "createSurface": {{"surfaceId": "login-1", "catalogId": "{catalog_id}", "sendDataModel": true}}}},
  {{"version": "v0.9", "updateComponents": {{"surfaceId": "login-1", "components": [
    {{"id": "root", "component": "Column", "children": ["title", "user", "pass", "submit"]}},
    {{"id": "title", "component": "Text", "text": "Login", "variant": "h2"}},
    {{"id": "user", "component": "TextField", "label": "Username", "value": {{"path": "/username"}}}},
    {{"id": "pass", "component": "TextField", "label": "Password", "value": {{"path": "/password"}}, "variant": "obscured"}},
    {{"id": "submit", "component": "Button", "child": "submit_label", "variant": "primary",
      "action": {{"event": {{"name": "login_submitted", "context": {{"user": {{"path": "/username"}}}}}}}}}},
    {{"id": "submit_label", "component": "Text", "text": "Sign In"}}
  ]}}}}
]
{A2UI_CLOSE_TAG}

### Example (templated list — note the RELATIVE paths inside the template)
{A2UI_OPEN_TAG}
[
  {{"version": "v0.9", "createSurface": {{"surfaceId": "list-1", "catalogId": "{catalog_id}", "sendDataModel": true}}}},
  {{"version": "v0.9", "updateDataModel": {{"surfaceId": "list-1", "path": "/", "value": {{"items": [
    {{"id": "1", "title": "First", "price": "$10"}},
    {{"id": "2", "title": "Second", "price": "$20"}}
  ]}}}}}},
  {{"version": "v0.9", "updateComponents": {{"surfaceId": "list-1", "components": [
    {{"id": "root", "component": "Column", "children": ["heading", "items_list"]}},
    {{"id": "heading", "component": "Text", "text": "Results", "variant": "h2"}},
    {{"id": "items_list", "component": "List", "children": {{"componentId": "item_card", "path": "/items"}}}},
    {{"id": "item_card", "component": "Card", "child": "item_row"}},
    {{"id": "item_row", "component": "Row", "justify": "spaceBetween", "children": ["item_title", "item_price", "pick"]}},
    {{"id": "item_title", "component": "Text", "text": {{"path": "title"}}}},
    {{"id": "item_price", "component": "Text", "text": {{"path": "price"}}}},
    {{"id": "pick", "component": "Button", "child": "pick_label",
      "action": {{"event": {{"name": "item_selected", "context": {{"itemId": {{"path": "id"}}}}}}}}}},
    {{"id": "pick_label", "component": "Text", "text": "Select"}}
  ]}}}}
]
{A2UI_CLOSE_TAG}
"""
    if client_capabilities:
        instructions += (
            "\n### Client capabilities\n"
            "The client reported these A2UI capabilities; only use what they allow:\n"
            f"{json.dumps(client_capabilities, ensure_ascii=False)}\n"
        )
    return instructions


def build_agui_instructions(agent_config: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Single entry point: returns the instruction block for an agent config,
    or None when A2UI is disabled. Used by BaseAgent._augment_message to
    inject and by the A2A server layer to strip echoes.
    """
    if not is_agui_enabled(agent_config):
        return None
    resolved = resolve_catalog(agent_config)
    return get_a2ui_system_instructions(
        catalog_id=resolved["catalog_id"],
        components_section=resolved["components_section"],
    )
