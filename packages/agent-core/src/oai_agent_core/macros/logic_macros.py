def macro_set(*args, **kwargs):
    """
    Macro to set a variable.
    Usage: {{ SET var_name value }}
    Returns the value.
    """
    if len(args) < 2:
        return "Error: SET macro requires var_name and value."
    
    var_name = args[0]
    value = " ".join(args[1:])
    
    context = kwargs.get('context', {})
    context[var_name] = value
    return value

def macro_get(*args, **kwargs):
    """
    Macro to get a variable.
    Usage: {{ GET var_name [default] }}
    """
    if not args:
        return "Error: GET macro requires a variable name."
    
    var_name = args[0]
    default_val = args[1] if len(args) > 1 else f"{{{{ GET {var_name} }}}}" 
    
    context = kwargs.get('context', {})
    return str(context.get(var_name, default_val))

def macro_if(*args, **kwargs):
    """
    Macro for conditional logic.
    Usage: {{ IF condition true_value false_value }}
    Condition is evaluated as a boolean (non-empty string, non-zero number, "true").
    """
    if len(args) < 3:
        return "Error: IF macro requires condition, true_value, and false_value."
    
    condition = args[0]
    true_val = args[1]
    false_val = args[2]
    
    # Determine truthiness
    is_true = False
    if isinstance(condition, str):
        if condition.lower() in ('true', 'yes', '1', 'on'):
            is_true = True
        elif condition.lower() in ('false', 'no', '0', 'off', ''):
            is_true = False
        else:
            # Non-empty string is true
            is_true = bool(condition)
    else:
        is_true = bool(condition)
        
    return true_val if is_true else false_val

def macro_loop(*args, **kwargs):
    """
    Macro to repeat a template string multiple times.
    Usage: {{ LOOP count "template" [separator] }}
    
    The template string can contain macros which will be evaluated for each iteration.
    Note: Inner braces in the template must be escaped or the template must be a string literal.
    
    Example: {{ LOOP 3 "User {{ RANDOM_INT 1 100 }}" ", " }}
    """
    if len(args) < 2:
        return "Error: LOOP macro requires count and template."
        
    try:
        count = int(args[0])
    except ValueError:
        return "Error: LOOP count must be an integer."
        
    template = args[1]
    separator = args[2] if len(args) > 2 else ""
    
    # Unescape delayed evaluation markers [[ ]] to {{ }}
    template = template.replace("[[", "{{").replace("]]", "}}")
    
    # Get context and processor
    context = kwargs.get('context', {})
    processor = kwargs.get('processor')
    
    if not processor:
        return "Error: LOOP macro requires processor instance."
    
    results = []
    for i in range(count):
        # Add iteration index to context temporarily
        context['LOOP_INDEX'] = str(i)
        context['LOOP_COUNT'] = str(i + 1)
        
        # Process macros in the template for this iteration
        # We need to recursively call process on the template
        processed_item = processor.process(template, context)
        results.append(processed_item)
        
    # Clean up context
    if 'LOOP_INDEX' in context:
        del context['LOOP_INDEX']
    if 'LOOP_COUNT' in context:
        del context['LOOP_COUNT']
        
    return separator.join(results)

def macro_repeat(*args, **kwargs):
    """
    Macro to repeat text multiple times.
    Usage: {{ REPEAT text count [separator] }}
    Default separator: empty string
    """
    if len(args) < 2:
        return "Error: REPEAT macro requires text and count."
    
    text = args[0]
    try:
        count = int(args[1])
    except ValueError:
        return "Error: count must be an integer."
    
    separator = args[2] if len(args) > 2 else ""
    
    return separator.join([text] * count)
