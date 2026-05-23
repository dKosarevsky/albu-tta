from __future__ import annotations

from dataclasses import dataclass

from learned_tta.teacher import TeacherBundle, load_teacher


@dataclass
class _FakeModel:
    eval_called: bool = False

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self


def test_load_teacher_uses_timm_model_config_and_eval(monkeypatch) -> None:
    fake_model = _FakeModel()
    calls: dict[str, object] = {}

    def fake_create_model(model_name: str, pretrained: bool) -> _FakeModel:
        calls["model_name"] = model_name
        calls["pretrained"] = pretrained
        return fake_model

    def fake_resolve_model_data_config(model: _FakeModel) -> dict[str, object]:
        calls["resolved_model"] = model
        return {"input_size": (3, 224, 224), "mean": (0.5, 0.5, 0.5)}

    def fake_create_transform(**kwargs):
        calls["transform_kwargs"] = kwargs
        return "preprocess"

    monkeypatch.setattr("learned_tta.teacher.timm.create_model", fake_create_model)
    monkeypatch.setattr(
        "learned_tta.teacher.timm.data.resolve_model_data_config",
        fake_resolve_model_data_config,
    )
    monkeypatch.setattr("learned_tta.teacher.timm.data.create_transform", fake_create_transform)

    bundle = load_teacher("resnet50.a1_in1k", pretrained=True)

    assert bundle == TeacherBundle(
        model=fake_model,
        data_config={"input_size": (3, 224, 224), "mean": (0.5, 0.5, 0.5)},
        preprocess="preprocess",
        model_name="resnet50.a1_in1k",
        pretrained=True,
    )
    assert fake_model.eval_called
    assert calls["model_name"] == "resnet50.a1_in1k"
    assert calls["pretrained"] is True
    assert calls["resolved_model"] is fake_model
    assert calls["transform_kwargs"] == {
        "input_size": (3, 224, 224),
        "mean": (0.5, 0.5, 0.5),
        "is_training": False,
    }
