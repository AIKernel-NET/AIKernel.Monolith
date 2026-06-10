from aikernel_monolith.docker_runner import select_image


def test_select_image_returns_monolith_tag():
    assert select_image().startswith("aikernel/monolith:")
