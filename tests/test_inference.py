import pytest

import numpy as np

import MouseArmTransformer

def _create_test_data(length):
    original_joints = MouseArmTransformer.helper.mausspaun_keys
    return {
        key : np.random.normal(size = (length, 2))
        for key in original_joints
    }

def test_import():
    assert hasattr(MouseArmTransformer, "run_inference")

@pytest.mark.parametrize("length", [8, 10, 100])
def test_run_inference(length):
    data = _create_test_data(length)
    lifted_data = MouseArmTransformer.run_inference(data)

    assert set(data.keys()) == set(lifted_data.keys())

    for key in lifted_data.keys():
        assert len(data[key].shape) == len(lifted_data[key].shape)
        assert len(lifted_data[key].shape) == 2
        assert data[key].shape[0] == lifted_data[key].shape[0]
        assert data[key].shape[1] == 2
        assert lifted_data[key].shape[1] == 3

def test_run_inference_missing_joint():
    data = _create_test_data(10)
    del data["right_shoulder"]
    with pytest.raises(KeyError):
        MouseArmTransformer.run_inference(data)

def test_run_inference_wrong_named_joint():
    pytest.skip("TODO: This is no longer handled properly.")
    data = _create_test_data(10)
    data["invalid_key"] = data["right_shoulder"]
    with pytest.raises(KeyError):
        MouseArmTransformer.run_inference(data)


@pytest.mark.parametrize("length,seq_length", [(10, 20), (5, 10), (7, 7)])
def test_run_inference_wrong_length(length, seq_length):
    data = _create_test_data(length)
    with pytest.raises(ValueError):
        MouseArmTransformer.run_inference(data, seq_length = seq_length)