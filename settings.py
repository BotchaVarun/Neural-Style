def get_version():
    try:
        # your existing version reading code
        return version_dict['__version__']
    except KeyError:
        return "0.1.0"  # fallback version