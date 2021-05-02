import pytest

from dynode import Recorder


class RecordableType:
    class RecordableSubtype:
        x = 1
        y = 2

    sub = RecordableSubtype()


def test_recorder_nonexisting():
    r = Recorder()
    obj = RecordableType()

    with pytest.raises(AttributeError):
        r.store(obj, "muppelimupp")


def test_recorder_without_alias():
    r = Recorder()

    obj = RecordableType()
    r.store(obj, "sub.x")

    for i in range(10):
        r(i, i)

    assert len(r[obj]["sub.x"]) == 10
    assert r[obj]["sub.x"] == [1] * 10

    assert len(r[obj]["time"]) == 10
    assert r[obj]["time"] == list(range(10))


def test_recorder_with_alias():
    r = Recorder()

    obj = RecordableType()
    r.store(obj, "sub.x", "x")

    for i in range(10):
        r(i, i)

    assert len(r[obj]["x"]) == 10
    assert r[obj]["x"] == [1] * 10

    assert len(r[obj]["time"]) == 10
    assert r[obj]["time"] == list(range(10))
