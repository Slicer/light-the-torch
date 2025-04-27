import contextlib
import functools
import importlib

import importlib.metadata as importlib_metadata
import inspect
import itertools

from unittest import mock

from pip._vendor.packaging.requirements import Requirement


class UnexpectedInternalError(Exception):
    def __init__(self, msg) -> None:
        actual_pip_version = Requirement(f"pip=={importlib_metadata.version('pip')}")
        required_pip_version = next(
            requirement
            for requirement in (
                Requirement(requirement_string)
                for requirement_string in importlib_metadata.requires("light_the_torch")
            )
            if requirement.name == "pip"
        )
        super().__init__(
            f"{msg}\n\n"
            f"This can happen when the actual pip version (`{actual_pip_version}`) "
            f"and the one required by light-the-torch (`{required_pip_version}`) "
            f"are out of sync.\n"
            f"If that is the case, please reinstall light-the-torch. "
            f"Otherwise, please submit a bug report at "
            f"https://github.com/Slicer/light-the-torch/issues"
        )


class Input(dict):
    def __init__(self, fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__fn__ = fn

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value) -> None:
        self[key] = value

    def __delattr__(self, key) -> None:
        del self[key]

    @classmethod
    def from_call_args(cls, fn, *args, **kwargs):
        params = iter(inspect.signature(fn).parameters.values())
        for arg, param in zip(args, params):
            kwargs[param.name] = arg
        for param in params:
            if (
                param.name not in kwargs
                and param.default is not inspect.Parameter.empty
            ):
                kwargs[param.name] = param.default
        return cls(fn, kwargs)

    def to_call_args(self):
        params = iter(inspect.signature(self.__fn__).parameters.values())

        args = []
        for param in params:
            if param.kind != inspect.Parameter.POSITIONAL_ONLY:
                break

            args.append(self[param.name])
        else:
            return (), dict()
        args = tuple(args)

        kwargs = dict()
        sentinel = object()
        for param in itertools.chain([param], params):
            kwarg = self.get(param.name, sentinel)
            if kwarg is not sentinel:
                kwargs[param.name] = kwarg

        return args, kwargs


@contextlib.contextmanager
def apply_fn_patch(
    *parts,
    preprocessing=lambda input: input,
    context=contextlib.nullcontext,
    postprocessing=lambda input, output: output,
):
    target = ".".join(parts)
    fn = import_obj(target)

    @functools.wraps(fn)
    def new(*args, **kwargs):
        input = Input.from_call_args(fn, *args, **kwargs)

        preprocessing(input)
        with context(input):
            args, kwargs = input.to_call_args()
            output = fn(*args, **kwargs)
        return postprocessing(input, output)

    with mock.patch(target, new=new):
        yield


def import_obj(target: str):
    attrs = []
    name = target
    while name:
        try:
            module = importlib.import_module(name)
            break
        except ImportError:
            try:
                name, attr = name.rsplit(".", 1)
            except ValueError:
                attr = name
                name = ""
            attrs.insert(0, attr)
    else:
        raise UnexpectedInternalError(
            f"Tried to import `{target}`, "
            f"but the top-level namespace `{attrs[0]}` doesn't seem to be a module."
        )

    obj = module
    for attr in attrs:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            raise UnexpectedInternalError(
                f"Failed to access `{attr}` from `{obj.__name__}`"
            ) from None

    return obj
