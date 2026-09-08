import pytest

from worker import registry


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    yield
    registry.clear()


def test_register_and_get_function():
    registry.register_function("square", lambda x: x * x)
    fn = registry.get_function("square")
    assert fn(5) == 25


def test_get_unregistered_function_returns_none():
    assert registry.get_function("missing") is None


def test_is_registered():
    assert not registry.is_registered("square")
    registry.register_function("square", lambda x: x * x)
    assert registry.is_registered("square")


def test_register_duplicate_name_raises_by_default():
    registry.register_function("square", lambda x: x * x)
    with pytest.raises(registry.FunctionAlreadyRegisteredError):
        registry.register_function("square", lambda x: x * x * x)


def test_register_duplicate_name_with_replace_true_overwrites():
    registry.register_function("square", lambda x: x * x)
    registry.register_function("square", lambda x: x * x * x, replace=True)
    assert registry.get_function("square")(2) == 8


def test_register_non_callable_raises():
    with pytest.raises(ValueError):
        registry.register_function("not_a_function", 42)


def test_register_empty_name_raises():
    with pytest.raises(ValueError):
        registry.register_function("", lambda: None)


def test_register_decorator():
    @registry.register("cube")
    def cube(x):
        return x * x * x

    assert registry.get_function("cube") is cube
    assert registry.get_function("cube")(3) == 27


def test_clear_removes_all_registrations():
    registry.register_function("square", lambda x: x * x)
    registry.clear()
    assert registry.get_function("square") is None
