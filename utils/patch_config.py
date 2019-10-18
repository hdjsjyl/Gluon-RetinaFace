import types
import inspect


class NoThrowBase:
    def __getattr__(self, item):
        return None


class NoThrowMeta(type):
    def __getattr__(self, item):
        return None


def patch_config_as_nothrow(instance):
    if "NoThrow" in [instance.__name__, instance.__class__.__name__]:
        return instance

    if type(instance) == type:
        instance = types.new_class(instance.__name__ + "NoThrow", (instance, ), dict(metaclass=NoThrowMeta))
        for (k, v) in inspect.getmembers(instance):
            if not k.startswith("__") and type(v) == type:
                type.__setattr__(instance, k, patch_config_as_nothrow(v))
    else:
        for (k, v) in inspect.getmembers(instance.__class__):
            if not k.startswith("__") and type(v) == type:
                type.__setattr__(instance.__class__, k, patch_config_as_nothrow(v))
        instance.__class__ = type(instance.__class__.__name__ + "NoThrow", (instance.__class__, NoThrowBase), {})

    return instance

if __name__ == 'main':
    ## testing section
    print('testing section')
