def test_no_object_skips_face_and_hand_processing():
    calls = {"face": 0, "hand": 0}

    def fake_food_detector_process(frame, clean_img):
        return []

    def fake_face_tracker_process(*args, **kwargs):
        calls["face"] += 1

    def fake_hand_tracker_process(*args, **kwargs):
        calls["hand"] += 1
        return []

    detected_food = fake_food_detector_process(None, None)
    if detected_food:
        fake_face_tracker_process(None, None, None, 0, 0)
        fake_hand_tracker_process(None, 0)

    assert detected_food == []
    assert calls == {"face": 0, "hand": 0}
