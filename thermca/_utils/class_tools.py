from functools import wraps


def freeze(cls):
    """Decorator prevents adding instance attributes at runtime

    Does not work for class attributes
    """

    cls.__frozen = False

    def frozensetattr(self, key, value):
        if self.__frozen and key not in dir(self):
            raise AttributeError(
                f"Cannot attach attribute '{key}' to frozen class '{cls.__name__}'."
            )
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True
        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls


class Frozen(type):
    """Metaclass prevents adding class attributes at runtime

    Does not work for instance attributes
    Does not work for derived classes

    Example::
        class Foo(metaclass=Frozen):
            b = 1
    """

    def __new__(cls, name, bases, dct):
        inst = super().__new__(cls, name, bases, {"_Frozen__frozen": False, **dct})
        inst.__frozen = True
        return inst

    def __setattr__(self, key, value):
        if self.__frozen and not hasattr(self, key):
            raise AttributeError(
                f"Cannot attach attribute '{key}' to frozen class '{self.__name__}'."
            )
        super().__setattr__(key, value)


if __name__ == '__main__':

    class Foo(metaclass=Frozen):
        b = 1

        def __init__(self, bar):
            self.bar = bar

    class Bar(Foo):
        b = 2


    Bar.b = 1


    foo = Foo(1)
    Foo.c = 3
    Foo.b = 42
    foo.bar = 42
    foo.foobar = "no way"
    print(Foo.b, foo.bar, foo.b)

    @freeze
    class Foo:
        b = 1

        def __init__(self, bar):
            self.bar = bar

    foo = Foo(1)
    Foo.c = 3
    Foo.b = 42
    foo.bar = 42
    foo.foobar = "no way"
    print(Foo.b, foo.bar, foo.b)