import uuid
import random
import json
import base64

def macro_uuid(*args, **kwargs):
    """
    Macro to generate a UUID.
    Usage: {{ UUID }}
    """
    return str(uuid.uuid4())

def macro_random_choice(*args, **kwargs):
    """
    Macro to pick a random item from a list.
    Usage: {{ RANDOM_CHOICE item1 item2 ... }}
    """
    if not args:
        return "Error: RANDOM_CHOICE macro requires at least one item."
    return random.choice(args)

def macro_random_int(*args, **kwargs):
    """
    Macro to generate a random integer.
    Usage: {{ RANDOM_INT min max }}
    """
    if len(args) != 2:
        return "Error: RANDOM_INT macro requires min and max arguments."
    
    try:
        min_val = int(args[0])
        max_val = int(args[1])
        return str(random.randint(min_val, max_val))
    except ValueError:
        return "Error: RANDOM_INT arguments must be integers."

def macro_faker(*args, **kwargs):
    """
    Macro to generate fake data using faker library.
    Usage: {{ FAKER provider_method [args...] }}
    """
    try:
        from faker import Faker
    except ImportError:
        return "Error: faker package is required. Install with 'pip install faker'."

    if not args:
        return "Error: FAKER macro requires a provider method name."

    # Check for locale in context
    context = kwargs.get('context', {})
    locale = context.get('FAKER_LOCALE', 'en_US')
    
    try:
        fake = Faker(locale)
    except Exception as e:
            return f"Error initializing Faker with locale '{locale}': {e}"

    method_name = args[0]
    method_args = args[1:]

    if not hasattr(fake, method_name):
        return f"Error: Faker has no method '{method_name}'."

    try:
        method = getattr(fake, method_name)
        converted_args = []
        for arg in method_args:
            try:
                converted_args.append(int(arg))
            except ValueError:
                try:
                    converted_args.append(float(arg))
                except ValueError:
                    converted_args.append(arg)
        
        return str(method(*converted_args))
    except Exception as e:
        return f"Error executing Faker method '{method_name}': {e}"

def macro_faker_locale(*args, **kwargs):
    """
    Macro to generate fake data with a specific locale.
    Usage: {{ FAKER_LOCALE locale provider_method [args...] }}
    Example: {{ FAKER_LOCALE fr_FR name }}
    """
    try:
        from faker import Faker
    except ImportError:
        return "Error: faker package is required. Install with 'pip install faker'."

    if len(args) < 2:
        return "Error: FAKER_LOCALE macro requires a locale and a provider method name."

    locale = args[0]
    method_name = args[1]
    method_args = args[2:]

    try:
        fake = Faker(locale)
    except Exception as e:
            return f"Error initializing Faker with locale '{locale}': {e}"

    if not hasattr(fake, method_name):
        return f"Error: Faker has no method '{method_name}'."

    try:
        method = getattr(fake, method_name)
        converted_args = []
        for arg in method_args:
            try:
                converted_args.append(int(arg))
            except ValueError:
                try:
                    converted_args.append(float(arg))
                except ValueError:
                    converted_args.append(arg)
        
        return str(method(*converted_args))
    except Exception as e:
        return f"Error executing Faker method '{method_name}': {e}"

def macro_base64(*args, **kwargs):
    """
    Macro to base64 encode a string.
    Usage: {{ BASE64 string }}
    """
    if not args:
        return ""
    
    text = " ".join(args)
    encoded_bytes = base64.b64encode(text.encode("utf-8"))
    return encoded_bytes.decode("utf-8")

def macro_json_escape(*args, **kwargs):
    """
    Macro to JSON escape a string.
    Usage: {{ JSON_ESCAPE string }}
    """
    if not args:
        return ""
    
    text = " ".join(args)
    return json.dumps(text)[1:-1]
