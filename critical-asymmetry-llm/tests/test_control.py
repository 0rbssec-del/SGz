from src.control.pid import PIDController, SlidingWindowA
from src.control.sampler import TemperatureTopPController
from src.control.multimodal_controller import MultiModalController


def test_pid_converges():
    pid = PIDController(A_star=0.5)
    for _ in range(100):
        pid.step(0.5)
    assert abs(pid.integral) < 1.0


def test_sliding_window():
    w = SlidingWindowA(window=4)
    for v in [0.1, 0.2, 0.3, 0.4]:
        avg = w.update(v)
    assert abs(avg - 0.25) < 1e-6


def test_sampler_predict():
    s = TemperatureTopPController()
    A = s.predict_A()
    assert 0.0 <= A <= 1.0


def test_multimodal_controller():
    c = MultiModalController()
    T, p, lv, tv = c.update(0.0)
    assert 0.5 <= T <= 1.5
    assert 0.5 <= p <= 0.99
