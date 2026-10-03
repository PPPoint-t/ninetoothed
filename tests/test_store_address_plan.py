from ninetoothed.backends.emitters.base import StoreAddressPlan
from ninetoothed.backends.emitters.cuda import CudaTarget
from ninetoothed.backends.emitters.ascend import AscendTarget
from ninetoothed.backends.emitters.triton import TritonTarget
from ninetoothed.backends.emitters import ssa


def test_default_targets_return_empty_generic_store_address_plan():
    for target in (CudaTarget(), TritonTarget()):
        plan = target.store_address_plan(
            value_name="value",
            tensor_info=None,
            level=0,
            context=None,
        )

        assert plan == StoreAddressPlan()
        assert plan.value_coords == ()
        assert plan.mask_coords == ()
        assert plan.source == "generic"


def test_ascend_store_address_hook_is_target_owned():
    assert AscendTarget.store_address_plan.__module__.endswith("emitters.ascend")


def test_missing_special_plan_uses_generic_empty_plan():
    plan = CudaTarget().store_address_plan(
        value_name="value", tensor_info=None, level=0, context=None
    )
    assert plan.value_coords == ()
    assert plan.mask_coords == ()


def test_illegal_store_address_plan_has_clear_contract_error():
    class InvalidTarget:
        def store_address_plan(self, **kwargs):
            del kwargs
            return "invalid"

    class Context:
        target = InvalidTarget()

    try:
        ssa._store_address_plan("value", None, 0, Context())
    except TypeError as exc:
        assert "StoreAddressPlan" in str(exc)
    else:
        raise AssertionError("invalid store plan was accepted")
