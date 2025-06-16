import json

class NoIndent:
    """ Wrapper class to mark lists that should not be indented """
    def __init__(self, value):
        self.value = value

class CustomEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles the NoIndent class to produce
    a compact string representation of the list
    """
    def default(self, obj):
        if isinstance(obj, NoIndent):
            # Return the value formatted as a compact string, without newlines
            return json.dumps(obj.value, separators=(',',':'))
        return super().default(obj)

