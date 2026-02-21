from .core import MacroProcessor
from .file_macros import (
    macro_open, macro_image, macro_path, macro_list_files, macro_concat, macro_sample,
    macro_json_extract, macro_template
)
from .data_macros import (
    macro_uuid, macro_random_choice, macro_random_int, macro_faker, macro_faker_locale,
    macro_base64, macro_json_escape
)
from .logic_macros import (
    macro_set, macro_get, macro_if, macro_loop, macro_repeat
)
from .time_macros import (
    macro_date, macro_now, macro_timestamp, macro_business_days_from
)
from .util_macros import (
    macro_env, macro_calc, macro_url_encode, macro_hash, macro_http_get, macro_sql_query,
    macro_truncate
)

# Register default macros
DEFAULT_MACROS = {
    "OPEN": macro_open,
    "IMAGE": macro_image,
    "PATH": macro_path,
    "LIST_FILES": macro_list_files,
    "CONCAT": macro_concat,
    "SAMPLE": macro_sample,
    "JSON_EXTRACT": macro_json_extract,
    "TEMPLATE": macro_template,
    
    "UUID": macro_uuid,
    "RANDOM_CHOICE": macro_random_choice,
    "RANDOM_INT": macro_random_int,
    "FAKER": macro_faker,
    "FAKER_LOCALE": macro_faker_locale,
    "BASE64": macro_base64,
    "JSON_ESCAPE": macro_json_escape,
    
    "SET": macro_set,
    "GET": macro_get,
    "IF": macro_if,
    "LOOP": macro_loop,
    "REPEAT": macro_repeat,
    
    "DATE": macro_date,
    "NOW": macro_now,
    "TIMESTAMP": macro_timestamp,
    "BUSINESS_DAYS_FROM": macro_business_days_from,
    
    "ENV": macro_env,
    "CALC": macro_calc,
    "URL_ENCODE": macro_url_encode,
    "HASH": macro_hash,
    "HTTP_GET": macro_http_get,
    "SQL_QUERY": macro_sql_query,
    "TRUNCATE": macro_truncate
}

# Update MacroProcessor to use these defaults if none provided
original_init = MacroProcessor.__init__

def new_init(self, project_root: str, macro_functions=None, logger=None):
    macros = DEFAULT_MACROS.copy()
    if macro_functions:
        macros.update(macro_functions)
    original_init(self, project_root, macros, logger)

MacroProcessor.__init__ = new_init
