def test_ner():
    from bkit.ner import Infer, visualize

    model = Infer("ner-noisy-label")
    text = "কাতার বিশ্বকাপে আর্জেন্টিনার বিশ্বকাপ জয়ে মার্তিনেজের অবদান অনেক।"
    predictions = model(text)
    visualize(predictions)

    # Assertions
    assert isinstance(predictions, list)
    for token, label, conf in predictions:
        assert isinstance(token, str)
        assert isinstance(label, str)
        assert 0 <= conf <= 1


def test_pos():
    from bkit.pos import Infer, visualize

    model = Infer("pos-noisy-label")
    text = "কাতার বিশ্বকাপে আর্জেন্টিনার বিশ্বকাপ জয়ে মার্তিনেজের অবদান অনেক।"
    predictions = model(text)
    visualize(predictions)

    # Assertions
    assert isinstance(predictions, list)
    for token, tag, conf in predictions:
        assert isinstance(token, str)
        assert isinstance(tag, str)
        assert 0 <= conf <= 1


def test_shallow():
    from bkit.shallow import Infer, visualize

    model = Infer("pos-noisy-label")
    text = "কাতার বিশ্বকাপে আর্জেন্টিনার বিশ্বকাপ জয়ে মার্তিনেজের অবদান অনেক।"
    predictions = model(text)
    # visualize(predictions)

    # Assertions
    assert isinstance(predictions, str)
    assert predictions.startswith("(S")


def test_dependency():
    from bkit.dependency import Infer, visualize

    model = Infer("dependency-parsing")
    text = "কাতার বিশ্বকাপে আর্জেন্টিনার বিশ্বকাপ জয়ে মার্তিনেজের অবদান অনেক।"
    predictions = model(text)
    visualize(predictions)

    # Assertions
    assert isinstance(predictions, list)
    assert "text" in predictions[0]
    assert "predictions" in predictions[0]
    for dep in predictions[0]["predictions"]:
        assert all(k in dep for k in ["token_start", "token_end", "label"])


def test_coref():
    from bkit.coref import Infer, visualize

    model = Infer("coref")
    text = (
        "তারাসুন্দরী ( ১৮৭৮ - ১৯৪৮ ) অভিনেত্রী । ১৮৮৪ সালে বিনোদিনীর সহায়তায় "
        "স্টার থিয়েটারে যোগদানের মাধ্যমে তিনি অভিনয় শুরু করেন । প্রথমে তিনি "
        "গিরিশচন্দ্র ঘোষের চৈতন্যলীলা নাটকে এক বালক ও সরলা নাটকে গোপাল চরিত্রে অভিনয় করেন ।"
    )
    predictions = model(text)
    visualize(predictions)

    # Assertions
    assert isinstance(predictions, dict)
    assert "text" in predictions
    assert "mention_indices" in predictions
    for mentions in predictions["mention_indices"].values():
        for span in mentions:
            assert "start_token" in span
            assert "end_token" in span
            assert isinstance(span["start_token"], int)
            assert isinstance(span["end_token"], int)
